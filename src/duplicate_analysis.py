
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "creditcard.csv")

features = df.drop(columns="Class").columns.tolist()

print("\n========== DUPLICATE ANALYSIS ==========")

# Exact duplicate rows
duplicate_mask = df.duplicated(keep=False)
duplicate_records = df[duplicate_mask]

print("\nRecords belonging to duplicate groups:")
print(len(duplicate_records))

print("\nDuplicate records by class:")
print(duplicate_records["Class"].value_counts())

# Repeated features, even if class labels differ
feature_duplicate_mask = df.duplicated(
    subset=features,
    keep=False
)

feature_duplicates = df[feature_duplicate_mask]

print("\nRecords with repeated feature values:")
print(len(feature_duplicates))

print("\nClass distribution among repeated features:")
print(feature_duplicates["Class"].value_counts())

# Check whether identical features have conflicting labels
conflicts = (
    feature_duplicates.groupby(features)["Class"]
    .nunique()
    .gt(1)
    .sum()
)

print("\nFeature groups with conflicting labels:")
print(conflicts)

print("\nDuplicate analysis completed!")
