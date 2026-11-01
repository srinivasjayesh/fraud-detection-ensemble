
import pandas as pd
import numpy as np
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

DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models"

MODEL_PATH.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Handle class imbalance
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Create models
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

# Train and evaluate
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

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, MODEL_PATH / filename)

# Save comparison results
results_df = pd.DataFrame(results)

results_df.to_csv(
    MODEL_PATH / "model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nAll models trained and saved successfully!")
