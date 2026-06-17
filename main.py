from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import json
import requests 
import httpx
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO,
                   format="[%(asctime)s] (line %(lineno)d) - %(levelname)s -%(message)s"
                   )

# ════════════════════════════════════════════════════════════════
# Sechema model for Response
# ════════════════════════════════════════════════════════════════
class WeatherItem(BaseModel):
    datetime: str
    temperature: float
    humidity: float
    windspeed: float
    cloudcover: float
class whether_response(BaseModel):
    status:str
    result:List[WeatherItem]
    
class ForecastItem(BaseModel):
    datetime: str
    block: int
    block_label: str
    predicted_load: float
    is_holiday: int
    holiday_name: str | None


class ForecastResponse(BaseModel):
    status: str
    forecast: list[ForecastItem]

# ── Load saved artifacts ───────────────────────────────────────
app = FastAPI(title="APU Power Forecast API", version="1.0.0")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cache storage
weather_cache = {
    "data":       None,
    "expires_at": None
}
# Load model and feature list once at startup
model       = joblib.load("xgboost_model.pkl")
feature_cols = json.load(open("feature_cols.json"))
history_df  = pd.read_csv("history.csv")
history_df['Datetime'] = pd.to_datetime(history_df['Datetime'])

logger.info(" Model loaded successfully")

FIXED_HOLIDAYS= {
    # Fixed date holidays (same every year)
    "2017-01-26": "Republic Day",
    "2017-08-15": "Independence Day",
    "2017-10-02": "Gandhi Jayanti",
    "2017-11-15": "Jharkhand Foundation Day",  # always Nov 15
    "2017-05-01": "Labour Day",                # always May 1
    "2017-12-25": "Christmas",                 # always Dec 25
    "2017-08-09":"World Tribal Day",
    "2017-04-14":"Good Friday",
}
LUNAR_HOLIDAYS_2017 = {
    # Lunar calendar holidays (2017 specific dates)
    "2017-01-14": "Makar Sankranti",
    "2017-03-13": "Holi",
    "2017-03-28": "Hindu new year",     
    "2017-03-30": "Sarhul",          # Jharkhand tribal
    "2017-04-05":"Ram Navami",
    "2017-04-14": "Ambedkar Jayanti",
    "2017-06-26": "Eid ul Fitr",
    "2017-08-07":"Raksha Bandhan",
    "2017-08-14": "Janmashtami",
    "2017-08-25":"Ganesh Chaturthi",
    "2017-09-01": "Eid ul Adha/Bakrid",
    "2017-09-17": "Karma Puja",      # Jharkhand local
    "2017-09-30": "Dussehra",
    "2017-10-01":"Muharram",
    "2017-10-19": "Diwali",
    "2017-10-20": "Diwali (2nd day)",
    "2017-10-26": "Chhath Puja",     # Biggest festival Jharkhand
    "2017-10-27": "Chhath Puja",     # 2 days celebrated
    "2017-11-04":"Guru Nanak Jayanti"
}
def is_holiday(date_str):
    """
    date_str format: 'YYYY-MM-DD'
    Returns holiday name or None
    """
    # Check fixed holidays first (MM-DD only)
    mm_dd = date_str[5:]   # extract MM-DD
    if mm_dd in FIXED_HOLIDAYS:
        return FIXED_HOLIDAYS[mm_dd]

    # Check lunar holidays (exact date match)
    if date_str in LUNAR_HOLIDAYS_2017:
        return LUNAR_HOLIDAYS_2017[date_str]

    return None


# ── Helper: Fetch forecast weather from Open-Meteo ────────────
async def fetch_forecast_weather(future_times):
    global weather_cache
     # Return cached data if still valid (cache for 30 minutes)
    if (weather_cache["data"] is not None and
        weather_cache["expires_at"] > datetime.now()):
        
        # Still merge with future_times
        future_df = pd.DataFrame({"Datetime": future_times})
        return future_df.merge(
            weather_cache["data"], on="Datetime", how="left"
        ).ffill().bfill()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get("https://api.open-meteo.com/v1/forecast", 
                params={
                "latitude":     23.7957,
                "longitude":    86.4304,
                "hourly":       ["temperature_2m", "relative_humidity_2m",
                             "windspeed_10m",  "cloudcover"],
                "forecast_days": 2
                },)
            r.raise_for_status()
        w = r.json()
        weather = pd.DataFrame({
            "Datetime":    pd.to_datetime(w["hourly"]["time"]),
            "Temperature": w["hourly"]["temperature_2m"],
            "Humidity":    w["hourly"]["relative_humidity_2m"],
            "WindSpeed":   w["hourly"]["windspeed_10m"],
            "CloudCover":  w["hourly"]["cloudcover"]
        })

        # Resample hourly → 10 min
        weather = (weather.set_index("Datetime")
                          .resample("10min")
                          .interpolate(method="linear")
                          .reset_index())
        
        # Save to cache — valid for 30 minutes
        weather_cache["data"]       = weather
        weather_cache["expires_at"] = datetime.now() + timedelta(minutes=30)

        # Match to future times
        future_df  = pd.DataFrame({"Datetime": future_times})
        weather_merged = future_df.merge(weather, on="Datetime", how="left")

        # Fill any remaining NaN with forward fill
        weather_merged = weather_merged.ffill().bfill()
        return weather_merged

    except Exception as e:
        logger.exception("Weather API failed")
        # Fallback: use seasonal averages for Dhanbad
        future_df = pd.DataFrame({"Datetime": future_times})
        future_df["Temperature"] = 25.0
        future_df["Humidity"]    = 65.0
        future_df["WindSpeed"]   = 2.0
        future_df["CloudCover"]  = 30.0
        return future_df


# ── Helper: Build feature dataframe for one block ─────────────
def build_features(pred_df, i, all_vals):
    row = pred_df.loc[[i]].copy()

    # Lag features
    row['Load_lag_1']    = all_vals[-1]
    row['Load_lag_3']    = all_vals[-3]    if len(all_vals) >= 3    else all_vals[-1]
    row['Load_lag_6']    = all_vals[-6]    if len(all_vals) >= 6    else all_vals[-1]
    row['Load_lag_12']   = all_vals[-12]   if len(all_vals) >= 12   else all_vals[-1]
    row['Load_lag_144']  = all_vals[-144]  if len(all_vals) >= 144  else all_vals[-1]
    row['Load_lag_288']  = all_vals[-288]  if len(all_vals) >= 288  else all_vals[-1]
    row['Load_lag_1008'] = all_vals[-1008] if len(all_vals) >= 1008 else all_vals[-1]

    # Rolling features
    s = pd.Series(all_vals)
    row['roll_mean_6']   = s.iloc[-6:].mean()
    row['roll_mean_12']  = s.iloc[-12:].mean()
    row['roll_mean_144'] = s.iloc[-144:].mean()
    row['roll_std_6']    = s.iloc[-6:].std() if len(all_vals) >= 6 else 0
    row['roll_max_144']  = s.iloc[-144:].max()
    row['roll_min_144']  = s.iloc[-144:].min()

    return row

import os

# Get the folder where main.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get('/dashboard')
def serve_dashboard():
    html_path = os.path.join(BASE_DIR, "dashboard.html")
    
    # Debug — print exact path being looked up
    print(f"Looking for HTML at: {html_path}")
    print(f"File exists: {os.path.exists(html_path)}")
    
    return FileResponse(html_path)
# ════════════════════════════════════════════════════════════════
# ENDPOINT 1: Health Check
# ════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status":  "running",
        "message": "APU Power Forecast API",
        "endpoints": ["/forecast", "/weather"]
    }

# ════════════════════════════════════════════════════════════════
# ENDPOINT 2: 24-hour forecast (144 blocks of 10 min)
# ════════════════════════════════════════════════════════════════
@app.get("/forecast",response_model=ForecastResponse)
async def get_forecast():
    try:
        # ── Step 1: Generate future timestamps ────────────────
        now          = pd.Timestamp.now().ceil("10min")
        future_times = pd.date_range(start=now, periods=144, freq="10min")

        # ── Step 2: Build prediction dataframe ────────────────
        pred_df = pd.DataFrame({"Datetime": future_times})

        # Time features
        pred_df['Block']     = pred_df['Datetime'].dt.hour * 6 + pred_df['Datetime'].dt.minute // 10
        pred_df['Hour']      = pred_df['Datetime'].dt.hour
        pred_df['DayOfWeek'] = pred_df['Datetime'].dt.dayofweek
        pred_df['IsWeekend'] = (pred_df['DayOfWeek'] >= 5).astype(int)
        pred_df['Month']     = pred_df['Datetime'].dt.month
        pred_df['Quarter']   = pred_df['Datetime'].dt.quarter

        # Cyclical encoding
        pred_df['Hour_sin']  = np.sin(2 * np.pi * pred_df['Hour']      / 24)
        pred_df['Hour_cos']  = np.cos(2 * np.pi * pred_df['Hour']      / 24)
        pred_df['Block_sin'] = np.sin(2 * np.pi * pred_df['Block']     / 144)
        pred_df['Block_cos'] = np.cos(2 * np.pi * pred_df['Block']     / 144)
        pred_df['Month_sin'] = np.sin(2 * np.pi * pred_df['Month']     / 12)
        pred_df['Month_cos'] = np.cos(2 * np.pi * pred_df['Month']     / 12)
        pred_df['DOW_sin']   = np.sin(2 * np.pi * pred_df['DayOfWeek'] / 7)
        pred_df['DOW_cos']   = np.cos(2 * np.pi * pred_df['DayOfWeek'] / 7)

        # ── Step 3: Weather features ──────────────────────────
        weather = await fetch_forecast_weather(future_times)
        pred_df['Temperature'] = weather['Temperature'].values
        pred_df['Humidity']    = weather['Humidity'].values
        pred_df['WindSpeed']   = weather['WindSpeed'].values
        pred_df['CloudCover']  = weather['CloudCover'].values

        # ── Step 4: Holiday features ──────────────────────────
        pred_df['holiday_name'] = pred_df['Datetime'].dt.strftime('%Y-%m-%d').map(is_holiday)
        pred_df['is_holiday']   = pred_df['holiday_name'].notna().astype(int)
        pred_df['holiday_name'] = pred_df['holiday_name'].where(
                                    pred_df['holiday_name'].notna(), other=None
)
        
        
        # ── Step 5: Rolling prediction loop ───────────────────
        #history  = list(history_df['Total_Load'].values[-1008:])
        all_vals = np.array(history_df['Total_Load'].values[-1008:], dtype=np.float64)

        predictions = []

        for i in range(144):
            #all_vals = history + predictions
            row      = build_features(pred_df, i, all_vals)
            pred_val = float(model.predict(row[feature_cols])[0])
            pred_val = max(0, pred_val)   # load can never be negative
            predictions.append(pred_val)
            all_vals = np.append(all_vals, pred_val)    

        # ── Step 6: Build response ────────────────────────────
        result = []
        for i in range(144):
            dt          = future_times[i]
            date_str    = dt.strftime('%Y-%m-%d')
            result.append(ForecastItem(
                datetime=       dt.strftime('%Y-%m-%d %H:%M'),
                block=         int(pred_df.loc[i, 'Block']),
                block_label=  dt.strftime('%H:%M'),
                predicted_load= round(predictions[i], 2),
                is_holiday=     int(pred_df.loc[i, 'is_holiday']),
                holiday_name=  pred_df.loc[i, "holiday_name"]
            ))

        return ForecastResponse (
            status="success",
            forecast=result
        )

    except Exception as e:
        logger.exception("forecast API failed")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )


# ════════════════════════════════════════════════════════════════
# ENDPOINT 3: Weather data for forecast period
# ════════════════════════════════════════════════════════════════
@app.get("/weather",response_model=whether_response)
async def get_weather():
    try:
        now          = pd.Timestamp.now().ceil("10min")
        future_times = pd.date_range(start=now, periods=144, freq="10min")
        weather_df      = await fetch_forecast_weather(future_times)

        weather = [
            WeatherItem(
                datetime=row["Datetime"].strftime("%Y-%m-%d %H:%M"),
                temperature=round(float(row["Temperature"]), 2),
                humidity=round(float(row["Humidity"]), 2),
                windspeed=round(float(row["WindSpeed"]), 2),
                cloudcover=round(float(row["CloudCover"]), 2),
            )
            for _, row in weather_df.iterrows()
        ]

        return whether_response (
            status="success",
            result=weather
        )

    except Exception as e:
        logger.exception("Weather API failed")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )

