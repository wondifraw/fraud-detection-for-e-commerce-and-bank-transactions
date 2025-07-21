import pandas as pd
import pytest
from src.imbalance_handling import ImbalanceHandler

def test_apply_smote():
    handler = ImbalanceHandler()
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 1, 0, 0]})
    y = pd.Series([0, 0, 1, 1])
    X_res, y_res = handler.apply_smote(X, y)
    assert len(X_res) == len(y_res)
    assert set(y_res) == {0, 1}

def test_apply_smote_edge():
    handler = ImbalanceHandler()
    X = pd.DataFrame({'a': [1], 'b': [1]})
    y = pd.Series([0])
    X_res, y_res = handler.apply_smote(X, y)
    assert X_res is None and y_res is None

def test_random_undersample():
    handler = ImbalanceHandler()
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'target': [0, 0, 1, 1]})
    result = handler.random_undersample(df, 'target')
    assert set(result['target']) == {0, 1}
    assert abs(result['target'].value_counts()[0] - result['target'].value_counts()[1]) <= 1

def test_random_undersample_edge():
    handler = ImbalanceHandler()
    df = pd.DataFrame({'a': [1], 'target': [0]})
    result = handler.random_undersample(df, 'target')
    assert result is not None 