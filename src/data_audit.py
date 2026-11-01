
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULT_PATH = ROOT / "results"

RESULT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\n========== DATA QUALITY ==========")

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Infinite Values:", np.isinf(df.select_dtypes(include="number")).sum().sum())

print("\n========== CLASS DISTRIBUTION ==========")

class_counts = df["Class"].value_counts()

print(class_counts)

class_distribution = pd.DataFrame({
    "Transaction Type": ["Genuine", "Fraud"],
    "Count": [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
})

class_distribution["Percentage"] = (
    class_distribution["Count"] / len(df) * 100
)

print(class_distribution.to_string(index=False))

print("\n========== AMOUNT ANALYSIS ==========")

print(df["Amount"].describe())

print("\n========== TIME ANALYSIS ==========")

print(df["Time"].describe())

print("\n========== FEATURE STATISTICS ==========")

feature_stats = df.drop(columns="Class").describe().T

feature_stats["Skewness"] = (
    df.drop(columns="Class").skew()
)

print(feature_stats.to_string())

feature_stats.to_csv(
    RESULT_PATH / "feature_statistics.csv"
)

print("\n========== OUTLIER ANALYSIS ==========")

outlier_results = []

for column in df.select_dtypes(include="number").columns:

    if column == "Class":
        continue

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (df[column] < lower) |
        (df[column] > upper)
    ).sum()

    outlier_results.append({
        "Feature": column,
        "Outlier Count": count
    })

outlier_df = pd.DataFrame(outlier_results)

print(
    outlier_df.sort_values(
        "Outlier Count",
        ascending=False
    ).to_string(index=False)
)

outlier_df.to_csv(
    RESULT_PATH / "outlier_analysis.csv",
    index=False
)

print("\nData quality audit completed!")
