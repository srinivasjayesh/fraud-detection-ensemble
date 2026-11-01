
# 🛡️ FraudGuard AI
### Credit Card Fraud Detection using Ensemble Learning

FraudGuard AI is a machine learning project that identifies potentially fraudulent credit card transactions using ensemble learning techniques.

The project explores data preprocessing, class imbalance handling, model comparison, threshold tuning, feature importance, and interactive fraud prediction through a Streamlit dashboard.

## 🚀 Live Application

[Open FraudGuard AI](YOUR_STREAMLIT_APP_URL)

## ✨ Features

- Interactive fraud detection dashboard
- XGBoost-based fraud probability prediction
- Adjustable classification threshold
- CSV transaction upload
- Fraud probability and risk classification
- Downloadable prediction results
- Transaction analytics and visualizations
- Model evaluation metrics
- Feature importance analysis
- Documentation and usage guide

## 🧠 Machine Learning Approach

The project uses the Kaggle Credit Card Fraud Detection dataset.

Original dataset:
- 284,807 transactions
- 492 fraudulent transactions
- 30 input features

After duplicate removal:
- 283,726 transactions
- 283,253 genuine transactions
- 473 fraudulent transactions

The dataset is highly imbalanced, so class imbalance was considered during model development.

### Models experimented with

- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- SMOTE-based experiments

### Selected model

XGBoost was selected based on validation performance and its balance between precision and recall.

The selected classification threshold is 0.70.

## 📊 Final Test Results

Evaluation was performed on an untouched stratified test set.

| Metric | Result |
|---|---:|
| Precision | 86.15% |
| Recall | 78.87% |
| F1 Score | 82.35% |
| ROC-AUC | 96.72% |
| PR-AUC | 81.31% |
| Accuracy | 99.94% |

Confusion matrix:

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 42,479 | 9 |
| Actual Fraud | 15 | 56 |

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## 📁 Project Structure

```text
fraud-detection-ensemble/
│
├── data/
├── models/
│   ├── v2/
│   └── [V1 models]
│
├── results/
│   ├── v2/
│   └── [V1 results]
│
├── src/
│   ├── data_audit.py
│   ├── duplicate_analysis.py
│   ├── clean_data.py
│   ├── train_v2.py
│   ├── smote_experiment.py
│   ├── xgboost_experiment.py
│   ├── xgboost_threshold_tuning.py
│   ├── final_evaluation.py
│   ├── feature_explainability.py
│   └── build_dashboard_assets.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/srinivasjayesh/fraud-detection-ensemble.git
cd fraud-detection-ensemble
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The deployed dashboard uses the saved model and precomputed dashboard assets. The original dataset is not required for normal application use.

## 📈 Explainability

The project uses XGBoost feature importance based on gain to identify features that contribute most to the model's tree splits.

Features V1–V28 are anonymized PCA-derived features and do not have directly interpretable business meanings.

## ⚠️ Limitations

- The dataset contains anonymized transaction features.
- The model has not been validated on live banking transactions.
- The evaluation uses a stratified random split and does not establish temporal generalization.
- Model predictions are estimates, not confirmed fraud findings.
- The application is an educational portfolio project, not a production financial security system.

## 👨‍💻 Author

Srinivas Jayesh Reddy

GitHub: https://github.com/srinivasjayesh

LinkedIn: https://linkedin.com/in/srinivas-jay-79617b343
