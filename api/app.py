from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow

app = Flask(__name__)

# Load the registered model from MLflow
MODEL_URI = "models:/iris-best-model/1"  # Change name & version as per your MLflow
model = mlflow.sklearn.load_model(MODEL_URI)

@app.route("/")
def home():
    return {"message": "Iris Model API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting JSON input: {"features": [5.1, 3.5, 1.4, 0.2]}
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)
        predicted_class = int(prediction[0])

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)