
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

_, X_test, _, y_test = __import__(
    "sklearn.model_selection",
    fromlist=["train_test_split"]
).train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

models = {
    "Random Forest": "random_forest.pkl",
    "AdaBoost": "adaboost.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

RESULT_PATH = ROOT / "results"
MODEL_PATH = ROOT / "models"

for name, filename in models.items():

    model = joblib.load(MODEL_PATH / filename)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=-1
    )

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance.importances_mean
    })

    feature_df = feature_df.sort_values(
        "Importance",
        ascending=False
    ).head(10)

    feature_df.to_csv(
        RESULT_PATH / f"{filename.replace('.pkl', '')}_features.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))

    plt.barh(
        feature_df["Feature"][::-1],
        feature_df["Importance"][::-1]
    )

    plt.title(f"{name} - Top 10 Important Features")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()

    plt.savefig(
        RESULT_PATH / f"{filename.replace('.pkl', '')}_importance.png"
    )

    plt.close()

print("Feature importance analysis completed!")
