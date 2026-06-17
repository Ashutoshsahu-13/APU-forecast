## APU Power Load Forecasting API
#### Overview

This project provides a 24-hour power load forecasting service using an XGBoost machine learning model. The API generates load forecasts at 10-minute intervals, integrates weather forecasts from Open-Meteo, and includes localized holiday information for visualization and analysis.

#### Features
- 24-hour load forecasting (144 blocks of 10 minutes)
- Weather forecast integration
- Holiday-aware forecasting
- FastAPI-based REST API
- Interactive Swagger documentation
- Dockerized deployment
- Dashboard for visualization

#### Project Structure
```text
app/
├── main.py   #Application Entry point
├── dashboard.html #dashboard UI
├── requirements.txt #Requirement file
├── Dockerfile #Docker container file
├── xgboost_model.pkl #Saved train model
├── feature_columns.pkl #saved features column which is used for training
├── history_data.csv #saved last 7 days data 
└── README.md #project documentation

```
#### Prerequisites
##### Local Development
- Python 3.13+
- pip
##### Docker Deployment
- Docker Desktop

#### Installation

#### Clone Repository
```text
git clone <repository-url>
cd apu-forecast
``` 
#### Create Virtual Environment
python -m venv venv

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

#### Install Dependencies
pip install -r requirements.txt

#### Running Locally

Start the FastAPI server:

uvicorn main:app --reload

API will be available at:

http://localhost:8000

Swagger Documentation:

http://localhost:8000/docs

Dashboard:

http://localhost:8000/dashboard

####    Docker Deployment
##### Build Docker Image
docker build -t apu-forecast:latest .
#### Run Docker Container
docker run -p 8000:8000 apu-forecast:latest

Verify container is running:

docker ps

Expected output:

0.0.0.0:8000->8000/tcp

#### Forecast Endpoint
GET /forecast

Returns:

- Forecast timestamp
- Predicted load
- Block number
- Holiday information

Example response:
```
{
  "status": "success",
  "forecast": [
    {
      "datetime": "2026-06-17 10:00",
      "block": 60,
      "block_label": "10:00",
      "predicted_load": 72500.12,
      "is_holiday": 0,
      "holiday_name": null
    }
  ]
}
```
#### Weather Endpoint
GET /weather

Returns:

- Temperature
- Humidity
- Wind Speed
- Cloud Cover

Example response:
```
{
  "status": "success",
  "result": [
    {
      "datetime": "2026-06-17 10:00",
      "temperature": 34.2,
      "humidity": 45.0,
      "windspeed": 7.5,
      "cloudcover": 20
    }
  ]
}
```

#### Model Information

Model: XGBoost Regressor

Features used:

- Time Features
- Cyclical Features
- Lag Features
- Rolling Statistics
- Weather Features
- Holiday Features

Performance Metrics:
```
MAE  : 396.73 W
RMSE : 699.94 W
R²   : 0.9980
MAPE : 0.61%
``` 

#### Weather Data

Open-Meteo Forecast API

Used Features:

- Temperature
- Humidity
- Wind Speed
- Cloud Cover
### Holiday Data

Indian national holidays and Jharkhand regional holidays.