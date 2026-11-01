
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "creditcard_cleaned.csv")

X = df.drop(columns="Class")
y = df["Class"]

# First split: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

splits = {
    "Training": y_train,
    "Validation": y_val,
    "Testing": y_test
}

print("\n========== V2 DATA SPLIT ==========")

for name, labels in splits.items():

    print(f"\n{name}")
    print("Total:", len(labels))
    print("Genuine:", (labels == 0).sum())
    print("Fraud:", (labels == 1).sum())
    print("Fraud Percentage:", round(labels.mean() * 100, 4), "%")

print("\nSplit completed successfully!")
