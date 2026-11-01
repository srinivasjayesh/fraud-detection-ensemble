
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

MODEL_PATH = ROOT / "models"
PLOT_PATH = ROOT / "results"

PLOT_PATH.mkdir(exist_ok=True)

model_names = [
    "random_forest",
    "adaboost",
    "gradient_boosting"
]

models = {}

# Load trained models
for name in model_names:
    models[name] = joblib.load(
        MODEL_PATH / f"{name}.pkl"
    )

# Model comparison chart
results = pd.read_csv(
    MODEL_PATH / "model_comparison.csv"
)

metrics = ["Precision", "Recall", "F1 Score", "PR-AUC"]

results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Ensemble Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(PLOT_PATH / "model_comparison.png")
plt.close()

# Confusion matrices
for name, model in models.items():

    predictions = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues"
    )

    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()

    plt.savefig(PLOT_PATH / f"{name}_confusion_matrix.png")
    plt.close()

# ROC and Precision-Recall curves
plt.figure(figsize=(9, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.plot(fpr, tpr, label=name.replace("_", " ").title())

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_PATH / "roc_comparison.png")
plt.close()

plt.figure(figsize=(9, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test, probabilities
    )

    plt.plot(recall, precision, label=name.replace("_", " ").title())

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_PATH / "precision_recall_comparison.png")
plt.close()

print("All visualizations generated successfully!")
