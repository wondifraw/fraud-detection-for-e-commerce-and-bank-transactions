import pandas as pd
import pytest
from src.data_split import separate_features_and_target, stratified_train_test_split

def test_separate_features_and_target():
    df = pd.DataFrame({'a': [1, 2], 'target': [0, 1]})
    X, y = separate_features_and_target(df, 'target')
    assert 'target' not in X.columns
    assert all(y == df['target'])

def test_separate_features_and_target_case_insensitive():
    df = pd.DataFrame({'A': [1, 2], 'Target': [0, 1]})
    X, y = separate_features_and_target(df, 'target')
    assert 'Target' not in X.columns
    assert all(y == df['Target'])

def test_separate_features_and_target_missing():
    df = pd.DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError):
        separate_features_and_target(df, 'target')

def test_stratified_train_test_split():
    X = pd.DataFrame({'a': range(10)})
    y = pd.Series([0]*5 + [1]*5)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 8 and len(X_test) == 2
    assert set(y_train) == {0, 1} and set(y_test) == {0, 1} 