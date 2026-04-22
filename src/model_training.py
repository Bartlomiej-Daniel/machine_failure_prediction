import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_model():
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"

    x_train = pd.read_csv(processed_dir / "x_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()

    ct_rf = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ])  
        ]
    )

    pipeline_rf = Pipeline([
        ('preprocessing', ct_rf),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training Random Forest model...")
    pipeline_rf.fit(x_train, y_train)
    
    print("Model learning completed successfully!")

    joblib.dump(pipeline_rf, base_dir / "models" / "rf_model.pkl")

if __name__ == "__main__":
    train_model()