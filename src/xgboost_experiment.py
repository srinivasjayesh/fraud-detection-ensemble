
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "creditcard_cleaned.csv"
MODEL_PATH = ROOT / "models" / "v2"
RESULT_PATH = ROOT / "results" / "v2"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns="Class")
y = df["Class"]

# Recreate the same split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# Calculate class imbalance ratio
scale_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Scale positive weight:", round(scale_weight, 2))

# Create XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

# Predict on validation data
probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Calculate metrics
precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

f1 = f1_score(
    y_val, predictions, zero_division=0
)

pr_auc = average_precision_score(
    y_val, probabilities
)

roc_auc = roc_auc_score(
    y_val, probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

print("\n========== XGBOOST RESULTS ==========")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print([[tn, fp], [fn, tp]])

# Save model
joblib.dump(
    model,
    MODEL_PATH / "xgboost.pkl"
)

# Save results
results = pd.DataFrame([{
    "Model": "XGBoost",
    "Threshold": 0.5,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc,
    "True Negatives": tn,
    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp
}])

results.to_csv(
    RESULT_PATH / "xgboost_results.csv",
    index=False
)

print("\nXGBoost model and results saved successfully!")
