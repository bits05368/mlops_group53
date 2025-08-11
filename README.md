# Iris ML Prediction API with MLOps Features

This project demonstrates a complete MLOps pipeline for an **Iris flower classification** model. It includes:

- Model training and experiment tracking with **MLflow**
- Model versioning and dynamic loading of the best model
- REST API built with **Flask** for predictions
- Input validation using **Pydantic**
- Request and prediction logging to file
- Prometheus-compatible metrics endpoint for monitoring
- Model retraining API endpoint
- Configuration via environment variables (`.env`)
- CI/CD pipeline with GitHub Actions
- Docker containerization for portability

---

## Architecture Overview
The system has the following components:

1. **MLflow** – for experiment tracking, model versioning, and model registry
2. **Flask API** – serves predictions and exposes retraining endpoints
3. **Prometheus Metrics** – `/metrics` endpoint for monitoring API performance
4. **Logging** – structured request & prediction logs stored in `logs/app.log`
5. **Dockerized Deployment** – consistent runtime environment
6. **CI/CD** – automated linting, testing, Docker build & push to Docker Hub

---

## 📂 Project Structure
```
├─ api/
│   └─ app.py              # Flask API with prediction, logging, monitoring, retraining
├─ data/
│   └─ iris.csv            # Iris dataset CSV
├─ src                     # Code files for model training and data preprocess
├─ requirements.txt        # Python dependencies
├─ Dockerfile              # Docker container instructions
├─ deploy.sh               # Deployment shell script
├─ .github/
│   └─ workflows/
│       └─ ci-cd.yml       # GitHub Actions workflow
├─ README.md
```

## 🚀 Features

1. **Experiment Tracking**  
   - Uses MLflow to log parameters, metrics, and models.
   - Automatically loads the best-performing model for predictions.

2. **REST API with Flask**  
   - `/predict`: Predicts Iris species from input features.
   - `/metrics`: Prometheus metrics for monitoring.
   - `/retrain`: Retrains the model using new data.

3. **Input Validation**  
   - Powered by Pydantic to ensure clean and valid inputs.

4. **Logging**  
   - Stores API request and prediction logs to a file.

5. **Monitoring**  
   - Prometheus-compatible `/metrics` endpoint.

6. **Containerization & CI/CD**  
   - Dockerized for consistent deployment.
   - GitHub Actions pipeline for linting, testing, and deployment.

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Create `.env` File
```env
MODEL_URI=mlruns/0/<model_id>/artifacts/model
LOG_FILE=prediction_requests.log
```

### 3️⃣ Run Locally
```bash
python api/app.py
```

### 4️⃣ Run with Docker
```bash
docker build -t iris-ml-api .
docker run -p 8000:8000 iris-ml-api
```


## 📈 CI/CD Pipeline
- **Lint & Test** on push to `dev` and `main` branches.
- **Build & Push** Docker image to Docker Hub.
- **Deploy** to local/EC2 environment.
