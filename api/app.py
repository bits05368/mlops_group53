import mlflow.pyfunc
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify


app = Flask(__name__)

# Set MLflow tracking URI to your local MLflow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_experiment")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Get experiment by name
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found")

# Search for the best run (highest accuracy)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])

if runs.empty:
    raise Exception("No runs found in experiment")

best_run_id = runs.iloc[0].run_id
MODEL_URI = f"runs:/{best_run_id}/model"
print(f"Loading model from best run: {best_run_id}")

# Load model from MLflow model registry (best run)
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded successfully!")

@app.route("/")
def home():
    return jsonify({"message": "Iris Prediction API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 8000))
    print(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port)