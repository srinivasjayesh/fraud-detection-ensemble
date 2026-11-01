
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "creditcard_cleaned.csv"
MODEL_PATH = ROOT / "models" / "v2"
RESULT_PATH = ROOT / "results" / "v2"

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

models = {
    "Random Forest + SMOTE": "random_forest_smote.pkl",
    "Gradient Boosting + SMOTE": "gradient_boosting_smote.pkl"
}

thresholds = np.arange(0.05, 1.00, 0.05)

all_results = []

for name, filename in models.items():

    print(f"\nTuning threshold for {name}...")

    model = joblib.load(MODEL_PATH / filename)

    probabilities = model.predict_proba(X_val)[:, 1]

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        all_results.append({
            "Model": name,
            "Threshold": round(threshold, 2),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp
        })

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULT_PATH / "threshold_comparison.csv",
    index=False
)

# Display best thresholds by F1
for name in models:

    model_results = results_df[
        results_df["Model"] == name
    ].sort_values("F1 Score", ascending=False)

    print(f"\n========== {name} ==========")
    print(
        model_results.head(5).to_string(index=False)
    )

print("\nThreshold comparison saved successfully!")
