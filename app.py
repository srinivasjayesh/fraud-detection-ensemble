
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models"
RESULT_PATH = ROOT / "results"

st.title("🛡️ FraudGuard AI")
st.caption("Intelligent Credit Card Fraud Detection using Ensemble Learning")

# Load models
@st.cache_resource
def load_models():

    return {
        "Random Forest": joblib.load(
            MODEL_PATH / "random_forest.pkl"
        ),
        "AdaBoost": joblib.load(
            MODEL_PATH / "adaboost.pkl"
        ),
        "Gradient Boosting": joblib.load(
            MODEL_PATH / "gradient_boosting.pkl"
        )
    }

models = load_models()

# Sidebar
st.sidebar.title("Settings")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

st.sidebar.info(
    "Select an ensemble model and adjust the threshold "
    "to control fraud detection sensitivity."
)

# Dashboard tabs
tab1, tab2, tab3 = st.tabs([
    "Fraud Detection",
    "Model Comparison",
    "Project Information"
])

# TAB 1: FRAUD DETECTION
with tab1:

    st.subheader("Transaction Analysis")

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        required_columns = list(
            models[selected_model].feature_names_in_
        )

        if set(data.columns) != set(required_columns):

            st.error(
                "Invalid dataset. Please upload a CSV "
                "with the original credit card transaction features."
            )

        else:

            data = data[required_columns]

            model = models[selected_model]

            probabilities = model.predict_proba(data)[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(int)

            results = data.copy()

            results["Fraud Probability"] = probabilities
            results["Prediction"] = [
                "Fraud" if p == 1 else "Genuine"
                for p in predictions
            ]

            total = len(results)
            fraud_count = int(predictions.sum())
            genuine_count = total - fraud_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", total)
            col2.metric("Fraud Detected", fraud_count)
            col3.metric("Genuine Transactions", genuine_count)

            st.subheader("Detection Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            st.subheader("Fraud Probability Distribution")

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.hist(probabilities, bins=30)

            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Number of Transactions")
            ax.set_title("Transaction Risk Distribution")

            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Download Prediction Results",
                data=results.to_csv(index=False),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

    else:

        st.info(
            "Upload a transaction CSV file to start fraud detection."
        )

# TAB 2: MODEL COMPARISON
with tab2:

    st.subheader("Ensemble Model Performance")

    comparison_path = MODEL_PATH / "model_comparison.csv"

    comparison = pd.read_csv(comparison_path)

    st.dataframe(
        comparison.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Performance Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    comparison.set_index("Model")[
        ["Precision", "Recall", "F1 Score", "PR-AUC"]
    ].plot(kind="bar", ax=ax)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Confusion Matrices
    st.subheader("Confusion Matrices")

    image_names = {
        "Random Forest": "random_forest_confusion_matrix.png",
        "AdaBoost": "adaboost_confusion_matrix.png",
        "Gradient Boosting": "gradient_boosting_confusion_matrix.png"
    }

    for name, filename in image_names.items():

        image_path = RESULT_PATH / filename

        st.write(f"**{name}**")

        if image_path.exists():
            st.image(
                str(image_path),
                use_container_width=True
            )
        else:
            st.warning(f"{filename} not found.")

    # ROC and Precision-Recall Curves
    st.subheader("ROC and Precision-Recall Curves")

    roc_path = RESULT_PATH / "roc_comparison.png"
    pr_path = RESULT_PATH / "precision_recall_comparison.png"

    if roc_path.exists():
        st.image(
            str(roc_path),
            caption="ROC Curve Comparison",
            use_container_width=True
        )

    if pr_path.exists():
        st.image(
            str(pr_path),
            caption="Precision-Recall Curve Comparison",
            use_container_width=True
        )

    # FEATURE IMPORTANCE
    st.subheader("Feature Importance Analysis")

    feature_images = {
        "Random Forest": "random_forest_importance.png",
        "AdaBoost": "adaboost_importance.png",
        "Gradient Boosting": "gradient_boosting_importance.png"
    }

    for name, filename in feature_images.items():

        st.write(f"**{name}**")

        image_path = RESULT_PATH / filename

        if image_path.exists():
            st.image(
                str(image_path),
                use_container_width=True
            )
        else:
            st.warning(
                f"{filename} not found. "
                "Run src/feature_analysis.py first."
            )

# TAB 3: PROJECT INFORMATION
with tab3:

    st.subheader("About FraudGuard AI")

    st.write("""
    FraudGuard AI is a machine learning based credit card
    fraud detection system using ensemble learning.

    It compares three classification algorithms:

    1. Random Forest
    2. AdaBoost
    3. Gradient Boosting

    The system identifies suspicious transactions and
    estimates their fraud probabilities.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    - Joblib
    """)

    st.subheader("Evaluation Metrics")

    st.write("""
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC
    """)

    st.subheader("Dataset Information")

    st.write("""
    Dataset: Credit Card Fraud Detection

    Total Transactions: 284,807

    Genuine Transactions: 284,315

    Fraudulent Transactions: 492
    """)
