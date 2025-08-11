# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirments.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirments.txt

# Copy all code
COPY . /app/

# Expose port your Flask app listens on
EXPOSE 8000

# Run the Flask app
CMD ["python", "api/app.py"]
