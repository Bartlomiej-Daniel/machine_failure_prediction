## Machine Failure Prediction

This project predicts machine failures using machine learning models trained on sensor data.
The goal is to detect potential failures early while balancing false alarms and missed failures.

## Demo

- Enter the machine parameters and see how the model predicts whether it will fail
- Link to demo: https://machinefailureprediction-amvl524ehvyfqbkmxcrquq.streamlit.app/

## Dataset

- Source: Kaggle Predictive Maintenance Dataset
- Size: ~10,000 samples
- Target: Machine failure (binary)
- Key features:
  - Torque
  - Rotational speed
  - Tool wear
  - Temperature

## Models

Three models were trained and compared:

- Logistic Regression
- Random Forest 
- Support Vector Machine (SVM)

## Results

Random Forest achieved the best balance:

- Precision: 0.90
- Recall: 0.63

Trade-off analysis:
- Logistic Regression → high recall, low precision
- SVM → high recall, low precision
- Random Forest → best balance

## Key Insights

- Torque, rotational speed, and tool wear are the most important factors
- Failures are strongly related to mechanical stress and wear
- Temperature also has a noticeable impact

## Limitations

- Imbalanced dataset (few failure cases)
- Limited ability to generalize rare events
- No time-series information (no machine history)