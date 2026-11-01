
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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

# Apply SMOTE only to training data
smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train
)

print("Training samples after SMOTE:")
print(y_train_smote.value_counts())

# Create boosting models
models = {
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    )
}

results = []

for name, model in models.items():

    print(f"\nTraining {name} with SMOTE...")

    # No additional class weights:
    # SMOTE has already balanced the training samples.
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

    results.append({
        "Model": name + " + SMOTE",
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
    })

    print("\nPrecision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])

    filename = name.lower().replace(" ", "_") + "_smote.pkl"
    joblib.dump(model, MODEL_PATH / filename)

# Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH / "smote_boosting_results.csv",
    index=False
)

print("\n========== SMOTE BOOSTING RESULTS ==========")
print(results_df.to_string(index=False))

print("\nExperiment completed successfully!")
