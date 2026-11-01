
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "creditcard_cleaned.csv")
MODEL_PATH = os.path.join(ROOT, "models", "v2", "xgboost.pkl")
RESULTS_DIR = os.path.join(ROOT, "results", "v2")

os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLD = 0.70
RANDOM_STATE = 42

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

# Recreate the exact 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print("Test dataset:", X_test.shape)
print("Fraud cases:", int(y_test.sum()))

# Load the already-trained XGBoost model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Apply selected threshold
predictions = (probabilities >= THRESHOLD).astype(int)

# Calculate metrics
metrics = {
    "Model": "XGBoost",
    "Threshold": THRESHOLD,
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1 Score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
    "PR-AUC": average_precision_score(y_test, probabilities)
}

print("\n========== FINAL TEST RESULTS ==========")

for metric, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

# Save metrics
pd.DataFrame([metrics]).to_csv(
    os.path.join(RESULTS_DIR, "final_test_metrics.csv"),
    index=False
)

with open(os.path.join(RESULTS_DIR, "final_test_metrics.json"), "w") as file:
    json.dump(metrics, file, indent=4)

# Save confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)
plt.title("XGBoost - Final Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "final_test_confusion_matrix.png"),
    dpi=300
)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {metrics['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Test ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "final_test_roc_curve.png"),
    dpi=300
)
plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, probabilities)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"XGBoost (AP = {metrics['PR-AUC']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final Test Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "final_test_pr_curve.png"),
    dpi=300
)
plt.close()

print("\nFinal evaluation completed.")
print("Results saved in results/v2/")
