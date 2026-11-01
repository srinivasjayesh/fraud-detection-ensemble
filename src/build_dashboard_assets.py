
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "creditcard_cleaned.csv")
IMPORTANCE_PATH = os.path.join(
    ROOT, "results", "v2", "xgboost_feature_importance.csv"
)
OUTPUT_PATH = os.path.join(
    ROOT, "results", "v2", "dashboard_data.json"
)

df = pd.read_csv(DATA_PATH)

genuine = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

# Amount distribution using log-scaled bins
amount_edges = np.linspace(
    0,
    np.log1p(df["Amount"].max()),
    31
)

genuine_amount, _ = np.histogram(
    np.log1p(genuine["Amount"]),
    bins=amount_edges
)

fraud_amount, _ = np.histogram(
    np.log1p(fraud["Amount"]),
    bins=amount_edges
)

amount_labels = np.expm1(
    (amount_edges[:-1] + amount_edges[1:]) / 2
)

# Time distribution: elapsed hours, not clock time
time_edges = np.linspace(
    df["Time"].min(),
    df["Time"].max(),
    25
)

genuine_time, _ = np.histogram(
    genuine["Time"],
    bins=time_edges
)

fraud_time, _ = np.histogram(
    fraud["Time"],
    bins=time_edges
)

time_labels = (
    (time_edges[:-1] + time_edges[1:]) / 2
) / 3600

# Feature importance
importance_df = pd.read_csv(IMPORTANCE_PATH)
importance_df = importance_df.head(15)

# Dashboard information
dashboard_data = {
    "total_transactions": int(len(df)),
    "genuine_transactions": int(len(genuine)),
    "fraud_transactions": int(len(fraud)),
    "fraud_percentage": float(
        len(fraud) / len(df) * 100
    ),
    "mean_amount": float(df["Amount"].mean()),
    "median_amount": float(df["Amount"].median()),
    "max_amount": float(df["Amount"].max()),
    "feature_names": [
        col for col in df.columns if col != "Class"
    ],
    "amount_distribution": {
        "labels": amount_labels.tolist(),
        "genuine": genuine_amount.tolist(),
        "fraud": fraud_amount.tolist()
    },
    "time_distribution": {
        "labels": time_labels.tolist(),
        "genuine": genuine_time.tolist(),
        "fraud": fraud_time.tolist()
    },
    "feature_importance": {
        "features": importance_df["Feature"].tolist(),
        "values": importance_df["Importance"].tolist()
    }
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as file:
    json.dump(dashboard_data, file, indent=4)

print("Dashboard assets created successfully!")
print("Total transactions:", len(df))
print("Fraud transactions:", len(fraud))
print("Saved:", OUTPUT_PATH)
