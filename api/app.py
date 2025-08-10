import mlflow.pyfunc
import mlflow
import pandas as pd
import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify,Response
from pydantic import BaseModel, ValidationError, confloat
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()
# Initialize logger
logging.basicConfig(
    filename='prediction_requests.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Prometheus request counter
REQUEST_COUNTER = Counter('total_requests', 'Total number of requests')

app = Flask(__name__)

# Set MLflow tracking URI to your local MLflow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_experiment")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load best model dynamically
def load_best_model():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    if runs_df.empty:
        raise Exception("No runs found in experiment")

    best_run_id = runs_df.iloc[0].run_id
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

model = load_best_model()
print("Model loaded successfully!")

# Pydantic schema for input validation
class IrisInput(BaseModel):
    sepal_length_cm: confloat(gt=0)
    sepal_width_cm: confloat(gt=0)
    petal_length_cm: confloat(gt=0)
    petal_width_cm: confloat(gt=0)

request_count = 0

@app.before_request
def before_request():
    global request_count
    request_count += 1

@app.route("/")
def home():
    return jsonify({"message": "Iris Prediction API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():    
    try:
        json_data = request.get_json()
        if not json_data:
            logger.warning("Empty input data received")
            return jsonify({"error": "No input data provided"}), 400
        
        logger.info(f"Received input: {json_data}")

        # Validate input
        input_obj = IrisInput(**json_data)

        # Prepare DataFrame with correct column names
        input_df = pd.DataFrame([{
            "sepal length (cm)": input_obj.sepal_length_cm,
            "sepal width (cm)": input_obj.sepal_width_cm,
            "petal length (cm)": input_obj.petal_length_cm,
            "petal width (cm)": input_obj.petal_width_cm
        }])

        prediction = model.predict(input_df)
        pred_list = prediction.tolist()

        # Log input and prediction
        logger.info(f"Request input: {json_data}")
        logger.info(f"Prediction output: {pred_list}")

        return jsonify({"prediction": pred_list}), 200
    
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve.errors()}")
        return jsonify({"error": ve.errors()}), 422
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        new_data = request.get_json()
        if not new_data or not isinstance(new_data, list):
            msg = "Input should be a list of training samples"
            logger.warning(msg)
            return jsonify({"error": msg}), 400

        df = pd.DataFrame(new_data)
        required_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"]
        if not all(col in df.columns for col in required_cols):
            msg = f"Missing required columns. Need: {required_cols}"
            logger.warning(msg)
            return jsonify({"error": msg}), 400

        X = df[required_cols[:-1]]
        y = df["target"]

        new_model = LogisticRegression(max_iter=200)
        new_model.fit(X, y)

        with mlflow.start_run():
            mlflow.log_param("retrained", True)
            mlflow.sklearn.log_model(new_model, "model")

        logger.info(f"Model retrained on {len(df)} samples and logged")

        return jsonify({"message": "Model retrained and logged successfully"}), 200

    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", 8000))
    print(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port)