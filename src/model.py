import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def load_data(csv_path="data/iris.csv"):
    df = pd.read_csv(csv_path)
    # "target" column should exist; if not, regenerate using sklearn's dataset
    if "target" not in df.columns:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train_and_evaluate(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)