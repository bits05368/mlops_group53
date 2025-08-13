import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import get_data_folder


def load_and_preprocess():
  
    data_dir = get_data_folder()
    """Load Iris dataset and split into train/test sets."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # Combine features + target
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Save datasets  
    # Save CSVs in project-root/data
    df.to_csv(os.path.join(data_dir, "iris.csv"), index=False)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print("✅ Data saved to data/train.csv and data/test.csv")