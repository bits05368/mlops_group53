import mlflow
import mlflow.sklearn
from model import load_data, train_and_evaluate, save_model

def main(data_path="data/iris.csv", experiment_name="Iris-Experiment"):
    # Load data
    X, y = load_data(data_path)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Params
        params = {"n_estimators": 100, "random_state": 42}
        mlflow.log_params(params)

        # Train
        model, acc = train_and_evaluate(X, y, params)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Save and log model
        save_model(model, "model.joblib")
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Model trained with accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()