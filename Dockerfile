# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirments and install
COPY requirments.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirments.txt

# Copy all code
COPY . /app/

# Expose port your Flask app listens on
EXPOSE 8000

# Set default MLflow Tracking URI for Ubuntu (will be overridden by docker-compose)
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000/

# Run Flask with Gunicorn + Uvicorn worker
CMD ["gunicorn", "api.app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
