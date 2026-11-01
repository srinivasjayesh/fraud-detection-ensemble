
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "v2", "xgboost.pkl")
DATA_PATH = os.path.join(ROOT, "data", "creditcard_cleaned.csv")
OUTPUT_DIR = os.path.join(ROOT, "results", "v2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Load feature names
df = pd.read_csv(DATA_PATH, nrows=1)
feature_names = [col for col in df.columns if col != "Class"]

# Get XGBoost gain importance
importance = model.get_booster().get_score(
    importance_type="gain"
)

importance_data = []

for feature, score in importance.items():
    index = int(feature[1:]) if feature.startswith("f") else None
    importance_data.append({
        "Feature": feature if feature in feature_names else feature_names[index],
        "Importance": score
    })

importance_df = pd.DataFrame(importance_data)
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Save all feature importance values
importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "xgboost_feature_importance.csv"),
    index=False
)

# Plot top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(9, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance (Gain)")
plt.ylabel("Feature")
plt.title("XGBoost - Top 15 Important Features")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "xgboost_feature_importance.png"),
    dpi=300
)

plt.close()

print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))

print("\nFeature importance analysis completed.")
