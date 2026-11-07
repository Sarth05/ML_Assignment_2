
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("Online Shoppers Purchasing Intention Prediction")
st.write(
    "An interactive dashboard to evaluate five machine learning classifiers."
)

# Model files
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Select model
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a classification algorithm",
    list(MODEL_FILES.keys())
)

# Upload test dataset
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing input features and Revenue target",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "Revenue" not in data.columns:
        st.error("The uploaded CSV must contain the Revenue target column.")

    else:
        model_path = os.path.join(
            "models",
            MODEL_FILES[selected_model]
        )

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")

        else:
            try:
                model = load_model(model_path)

                expected_features = list(model.feature_names_in_)

                missing_features = [
                    feature for feature in expected_features
                    if feature not in data.columns
                ]

                if missing_features:
                    st.error(f"Missing features: {missing_features}")

                else:
                    X = data[expected_features]
                    y = data["Revenue"].astype(int)

                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]

                    # Calculate evaluation metrics
                    accuracy = accuracy_score(y, predictions)
                    auc = (
                        roc_auc_score(y, probabilities)
                        if y.nunique() == 2 else None
                    )
                    precision = precision_score(
                        y, predictions, zero_division=0
                    )
                    recall = recall_score(
                        y, predictions, zero_division=0
                    )
                    f1 = f1_score(
                        y, predictions, zero_division=0
                    )
                    mcc = matthews_corrcoef(y, predictions)

                    # Display metrics
                    st.subheader("Model Performance")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric(
                        "AUC",
                        f"{auc:.4f}" if auc is not None else "N/A"
                    )
                    col3.metric("Precision", f"{precision:.4f}")

                    col4, col5, col6 = st.columns(3)
                    col4.metric("Recall", f"{recall:.4f}")
                    col5.metric("F1 Score", f"{f1:.4f}")
                    col6.metric("MCC", f"{mcc:.4f}")

                    # Confusion matrix
                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y, predictions, labels=[0, 1]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["No Purchase", "Purchase"],
                        yticklabels=["No Purchase", "Purchase"],
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Classification report
                    st.subheader("Classification Report")

                    report = classification_report(
                        y,
                        predictions,
                        labels=[0, 1],
                        target_names=["No Purchase", "Purchase"],
                        output_dict=True,
                        zero_division=0
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4)
                    )

                    st.success(
                        f"Evaluation completed using {selected_model}."
                    )

            except Exception as error:
                st.error(f"Evaluation failed: {error}")

else:
    st.info("Please upload your test CSV file to evaluate a model.")

st.caption("Machine Learning Assignment 2 | Model Evaluation Dashboard")
