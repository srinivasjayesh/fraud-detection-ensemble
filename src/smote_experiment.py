
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

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

print("Before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE only to training data
smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest with SMOTE...")

model.fit(X_train_smote, y_train_smote)

# Evaluate on untouched validation data
probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

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

print("\n========== SMOTE RESULTS ==========")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print([[tn, fp], [fn, tp]])

# Save results
results = pd.DataFrame([{
    "Model": "Random Forest + SMOTE",
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
    RESULT_PATH / "smote_results.csv",
    index=False
)

joblib.dump(
    model,
    MODEL_PATH / "random_forest_smote.pkl"
)

print("\nSMOTE experiment completed successfully!")
