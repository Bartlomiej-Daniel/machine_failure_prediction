import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def preprocess_data():
    base_dir = Path(__file__).resolve().parent.parent
    raw_data = base_dir / "data" / "raw"
    processed_data = base_dir / "data" / "processed"

    df = pd.read_csv(raw_data / "predictive_maintenance.csv")

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[features]
    y = df["Target"]


    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    x_train.to_csv(processed_data / "x_train.csv", index=False)
    x_test.to_csv(processed_data / "x_test.csv", index=False)
    y_train.to_csv(processed_data / "y_train.csv", index=False)
    y_test.to_csv(processed_data / "y_test.csv", index=False)

    print("Data preprocessing completed and files saved!")

if __name__ == "__main__":
    preprocess_data()