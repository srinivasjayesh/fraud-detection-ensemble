
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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

# Create training, validation and testing sets
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

# Calculate class weights using training data only
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Create baseline models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

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

# Train and evaluate using validation data
for name, model in models.items():

    print(f"\nTraining {name}...")

    if name == "Random Forest":
        model.fit(X_train, y_train)
    else:
        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )

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

    roc_auc = roc_auc_score(y_val, probabilities)

    pr_auc = average_precision_score(y_val, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": name,
        "Threshold": 0.5,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, MODEL_PATH / filename)

# Save validation comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH / "baseline_validation.csv",
    index=False
)

print("\n========== V2 BASELINE COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nV2 baseline models trained and saved!")
