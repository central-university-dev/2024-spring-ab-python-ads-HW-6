import pytest
import pandas as pd
from main import get_classifier, train_and_evaluate
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Provides a sample dataset for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data including client_id, a feature column,
                      target (outcome), and treatment_flg (treatment indicator).
    """
    data = {
        'client_id': [1, 2, 3, 4, 5],
        'feature_1': [1, 2, 3, 4, 5],  # Note: Changed to numerical to simulate real features
        'target': [1, 0, 1, 0, 1],
        'treatment_flg': [1, 1, 0, 0, 1]
    }
    return pd.DataFrame(data)

def test_get_classifier() -> None:
    """
    Tests the get_classifier function to ensure it returns the correct classifier types
    based on the input string.
    """
    assert isinstance(get_classifier("CatBoostClassifier"), CatBoostClassifier), "CatBoostClassifier instance expected."
    assert isinstance(get_classifier("RandomForestClassifier"), RandomForestClassifier), "RandomForestClassifier instance expected."

def test_train_and_evaluate(sample_data: pd.DataFrame) -> None:
    """
    Tests the train_and_evaluate function with both "Solo Model" and "Two Model" approaches.
    Ensures that the function returns a float value representing the AUUC score for each model type.

    Args:
        sample_data (pd.DataFrame): A fixture providing a sample dataset for testing.
    """
    X_train = sample_data[['feature_1']]
    y_train = sample_data['target']
    treatment_train = sample_data['treatment_flg']

    # Duplicate the training data for testing due to the small sample size.
    X_test, y_test, treatment_test = X_train, y_train, treatment_train

    # Test for Solo Model with CatBoostClassifier
    auuc_score_solo = train_and_evaluate(
        "Solo Model", "CatBoostClassifier", X_train, X_test, y_train, y_test, treatment_train, treatment_test
    )
    assert isinstance(auuc_score_solo, float), "AUUC score for Solo Model should be a float."

    # Test for Two Model with RandomForestClassifier
    auuc_score_two = train_and_evaluate(
        "Two Model", "RandomForestClassifier", X_train, X_test, y_train, y_test, treatment_train, treatment_test
    )
    assert isinstance(auuc_score_two, float), "AUUC score for Two Model should be a float."
