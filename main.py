"""Main application for Uplift Modeling Dashboard.

This application integrates data processing, exploratory data analysis (EDA), and model training
and evaluation for uplift modeling using Streamlit for the user interface.
"""

import streamlit as st
from sklift.models import SoloModel, TwoModels
from sklift.metrics import qini_auc_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

# Load the datasets
uplift_train_df: pd.DataFrame = pd.read_csv('uplift_train.csv')
clients_df: pd.DataFrame = pd.read_csv('clients.csv')

# Merge the training data with client information
train_data_merged: pd.DataFrame = pd.merge(uplift_train_df, clients_df, on='client_id', how='left')

# Implement random selection of 80% of the dataset for training to simulate data updates
train_data_sample: pd.DataFrame = train_data_merged.sample(frac=0.8)

# Frequency Encoding for categorical variables
for col in train_data_sample.select_dtypes(include=['object', 'category']).columns:
    if col not in ['client_id']:
        freq: pd.Series = train_data_sample.groupby(col).size() / len(train_data_sample)
        train_data_sample[col + '_freq'] = train_data_sample[col].map(freq)
        train_data_sample.drop(col, axis=1, inplace=True)

# Streamlit UI setup
st.title("Uplift Modeling Dashboard")

# Selection of modeling approaches and classifiers
model_approaches: List[str] = st.multiselect(
    "Choose one or more uplift modeling approaches:", ["Solo Model", "Two Model"], default=["Solo Model"]
)
classifier_choices: List[str] = st.multiselect(
    "Choose one or more classifiers:", ["CatBoostClassifier", "RandomForestClassifier"], default=["CatBoostClassifier"]
)

def get_classifier(classifier_name: str) -> Union[CatBoostClassifier, RandomForestClassifier]:
    """Fetches the classifier based on the classifier name.

    Args:
        classifier_name (str): The name of the classifier.

    Returns:
        Union[CatBoostClassifier, RandomForestClassifier]: An instance of the requested classifier.
    """
    if classifier_name == "CatBoostClassifier":
        return CatBoostClassifier(verbose=0, thread_count=2)
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier()

def train_and_evaluate(
    model_approach: str, 
    classifier_name: str, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    treatment_train: pd.Series, 
    treatment_test: pd.Series
) -> float:
    """Trains and evaluates the uplift model based on the selected approach and classifier.

    Args:
        model_approach (str): The uplift modeling approach ("Solo Model" or "Two Model").
        classifier_name (str): The name of the classifier to use.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
        treatment_train (pd.Series): Training treatment indicator.
        treatment_test (pd.Series): Testing treatment indicator.

    Returns:
        float: The AUUC score for the evaluated model.
    """
    if model_approach == "Two Model":
        estimator_trmnt = get_classifier(classifier_name)
        estimator_ctrl = get_classifier(classifier_name)
        model = TwoModels(estimator_trmnt=estimator_trmnt, estimator_ctrl=estimator_ctrl, method='vanilla')
    else:
        estimator = get_classifier(classifier_name)
        model = SoloModel(estimator)

    model.fit(X_train, y_train, treatment_train)
    uplift_pred = model.predict(X_test)
    auuc_score = qini_auc_score(y_test, uplift_pred, treatment_test)
    return auuc_score

# EDA: Visualizations for Treatment Flag and Target Variable distributions
st.header("Exploratory Data Analysis")
st.subheader("Distribution of Treatment Flag")
fig, ax = plt.subplots()
sns.countplot(x='treatment_flg', data=train_data_sample, ax=ax)
st.pyplot(fig)

st.subheader("Distribution of Target Variable")
fig, ax = plt.subplots()
sns.countplot(x='target', data=train_data_sample, ax=ax)
st.pyplot(fig)

# Model Training and Evaluation
st.header("Model Training and Evaluation")
if st.button("Train and Evaluate Models"):
    with st.spinner('Model training in progress...'):
        # Extract features and target variables
        X: pd.DataFrame = train_data_sample.drop(['client_id', 'target', 'treatment_flg'], axis=1)
        y: pd.DataFrame = train_data_sample['target']
        treatment: pd.DataFrame = train_data_sample['treatment_flg']
        X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(X, y, treatment, test_size=0.3)
        
        results: List[Tuple[str, str, float]] = []
        for model_approach in model_approaches:
            for classifier_name in classifier_choices:
                auuc_score = train_and_evaluate(model_approach, classifier_name, X_train, X_test, y_train, y_test, treatment_train, treatment_test)
                results.append((model_approach, classifier_name, auuc_score))
                
        st.success("Model training completed.")
        # Display AUUC Scores for comparison
        st.subheader("Model Comparison by AUUC Scores")
        for result in results:
            st.write(f"{result[0]} + {result[1]}: AUUC Score = {result[2]:.4f}")
