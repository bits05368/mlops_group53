import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
from utils import get_data_folder
from datapreprocess import load_and_preprocess  

# Paths
datafolder = get_data_folder()
TRAIN_FP = os.path.join(datafolder, "train.csv")
TEST_FP = os.path.join(datafolder, "test.csv")

def load_data():
    train_df = pd.read_csv(TRAIN_FP)
    test_df = pd.read_csv(TEST_FP)
    # X: all numeric features except target columns
    X_train = train_df.drop(columns=["target", "target_name"], errors="ignore")
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target", "target_name"], errors="ignore")
    y_test = test_df["target"]
    return X_train, X_test, y_train, y_test

def train_and_log_model(model_cls, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print(f"{model_cls.__name__} accuracy: {acc:.4f}")
        return acc, mlflow.active_run().info.run_id

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change if needed
    mlflow.set_experiment("iris_experiment")
    load_and_preprocess()
    X_train, X_test, y_train, y_test = load_data()
    results = []
    print(f"start: logistic regression and random forest")
    # Model 1: Logistic Regression
    lr_params = {"max_iter": 200}
    acc, run_id = train_and_log_model(LogisticRegression, lr_params,
                                      X_train, X_test, y_train, y_test)
    results.append(("LogisticRegression", acc, run_id))
    # Model 2: Random Forest
    rf_params = {"n_estimators": 100, "random_state": 42}
    acc, run_id = train_and_log_model(RandomForestClassifier, rf_params,
                                      X_train, X_test, y_train, y_test)
    results.append(("RandomForestClassifier", acc, run_id))
    print(f"complete: logistic regression and random forest")
    # Pick best model
    best_model_name, best_acc, best_run_id = max(results, key=lambda x: x[1])
    print(f"Best model: {best_model_name} (acc={best_acc:.4f})")
    # -----------------------------
    # Register the best model
    # -----------------------------
    tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme
    model_uri = f"runs:/{best_run_id}/model"
    if tracking_uri_type != "file":
        # Register to MLflow Model Registry
        model_version = mlflow.register_model(model_uri, f"{best_model_name}_Model")
        print(f"Registered {best_model_name} as version {model_version.version} in MLflow Model Registry")
    else:
        print("Model Registry not supported with file-based tracking URI.")

if __name__ == "__main__":
    main()
    