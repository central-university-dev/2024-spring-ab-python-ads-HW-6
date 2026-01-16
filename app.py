import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklift.models import SoloModel, TwoModels
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklift.metrics import qini_auc_score
import seaborn as sns


uplift_train = pd.read_csv('uplift_train.csv')
clients_df = pd.read_csv('clients.csv')
train_merged = pd.merge(uplift_train, clients_df, on='client_id', how='left')
train_sampled = train_merged.sample(frac=0.8)

st.title("Uplift dashboard")

model_approaches = st.multiselect(
    "Select uplift modeling approaches:", ['Solo Model', 'Two Model'], default=['Solo Model']
)

classifier_choices = st.multiselect(
    "Select classifiers:", ['CatBoostClassifier', 'RandomForestClassifier'], default=['CatBoostClassifier']
)

def fetch_classifier(classifier_name):
    """Obtains the classifier based on its name."""

    if classifier_name == "CatBoostClassifier":
        return CatBoostClassifier(verbose=0)
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier()

def train_and_assess_model(
    approach, 
    classifier, 
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    treatment_train, 
    treatment_test
):
    """Trains and evaluates the uplift model using the chosen approach and classifier."""
    if approach == "Two Model":
        estimator_trmnt = fetch_classifier(classifier)
        estimator_ctrl = fetch_classifier(classifier)
        model = TwoModels(estimator_trmnt=estimator_trmnt, estimator_ctrl=estimator_ctrl, method='vanilla')
    else:
        estimator = fetch_classifier(classifier)
        model = SoloModel(estimator)

    model.fit(X_train, y_train, treatment_train)
    uplift_predictions = model.predict(X_test)
    auuc_score = qini_auc_score(y_test, uplift_predictions, treatment_test)
    return auuc_score

#Train_val
st.title("Model train and eval")

if st.button("Initiate model training and evaluation"):
    with st.spinner('Training models...'):
        X_data = train_sampled.drop(['client_id', 'target', 'treatment_flg'], axis=1)
        y_data = train_sampled['target']
        treat_data = train_sampled['treatment_flg']
        X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(X_data, y_data, treat_data, test_size=0.2)
        
        model_results = []
        for method in model_approaches:
            for classifier in classifier_choices:
                auuc_score = train_and_evaluate(method, classifier, X_train, X_test, y_train, y_test, treat_train, treat_test)
                model_results.append((method, classifier, auuc_score))
                
        st.success("Model training completed.")
        st.subheader("Model comparison based on AUUC scores")
        for result in model_results:
            st.write(f"Approach: {result[0]} - Classifier: {result[1]} - AUUC Score: {result[2]}")

# EDA
st.header("EDA")
st.subheader("treatment_flg distribution")
fig, ax = plt.subplots()
sns.countplot(x='treatment_flg', data=train_sampled, ax=ax)
st.pyplot(fig)

st.subheader("target distribution")
fig, ax = plt.subplots()
sns.countplot(x='target', data=train_sampled, ax=ax)
st.pyplot(fig)