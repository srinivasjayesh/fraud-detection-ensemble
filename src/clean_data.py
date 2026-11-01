
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "creditcard.csv"
OUTPUT_PATH = ROOT / "data" / "creditcard_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)

# Count duplicate rows before removal
duplicate_rows = df[df.duplicated(keep="first")]

print("\nDuplicate copies to remove:")
print(len(duplicate_rows))

print("\nDuplicate copies by class:")
print(duplicate_rows["Class"].value_counts())

# Remove exact duplicate rows
cleaned_df = df.drop_duplicates(keep="first")

print("\nCleaned dataset:", cleaned_df.shape)

print("\nNew class distribution:")
print(cleaned_df["Class"].value_counts())

print("\nRemaining duplicates:", cleaned_df.duplicated().sum())

# Save cleaned dataset
cleaned_df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
