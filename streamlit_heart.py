import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Title of the app
st.title("Heart Disease Prediction App")

# Load the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")

    # Display dataset
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Data preprocessing
    st.subheader("Data Preprocessing")
    st.write("Handling missing values...")
    data.fillna(data.median(numeric_only=True), inplace=True)
    st.write("Missing values handled!")

    # Handle categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))  # Convert strings to numbers

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Data visualization
    st.subheader("Data Visualization")

    # Pie chart for General_Health
    if "General_Health" in data.columns:
        fig1, ax1 = plt.subplots()
        data['General_Health'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Paired'), ax=ax1)
        ax1.set_title('General Health')
        ax1.set_ylabel('')
        st.pyplot(fig1)

    # Histogram for Weight
    if "Weight_(kg)" in data.columns:
        fig2, ax2 = plt.subplots()
        ax2.hist(data['Weight_(kg)'], color='green', edgecolor='black', bins=10)
        ax2.set_title('Histogram of Weight (kg)')
        ax2.set_xlabel('Weight (kg)')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

    # Box Plot for Weight by General Health
    if "General_Health" in data.columns and "Weight_(kg)" in data.columns:
        fig3, ax3 = plt.subplots()
        sns.boxplot(x=data['General_Health'], y=data['Weight_(kg)'], ax=ax3)
        ax3.set_title('Box Plot of Weight (kg) by General Health')
        st.pyplot(fig3)

    # Model Training Section
    st.subheader("Model Training")

    # Select features and target
    st.write("Selecting features and target...")
    columns = list(data.columns)
    features = st.multiselect("Select feature columns", columns)
    target = st.selectbox("Select target column", columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Ensure all features are numeric
        if X.select_dtypes(include=['object']).shape[1] > 0:
            st.error("Please ensure all selected features are numeric.")
        else:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Random Forest Classifier
            st.write("Training the model...")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            st.subheader("Model Metrics")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
            st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
            st.write("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

            # Save the model
            st.write("Saving the model...")
            joblib.dump(model, "heart_disease_model.pkl")
            st.success("Model saved as 'heart_disease_model.pkl'")
    else:
        st.warning("Please select at least one feature and the target column!")
else:
    st.info("Awaiting CSV file to be uploaded!")
