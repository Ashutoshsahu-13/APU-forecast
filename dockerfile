# Use the official Python 3.13.5 slim image as the base
FROM python:3.13.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]