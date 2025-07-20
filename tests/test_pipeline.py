import pandas as pd
import pytest

from src.data_loading import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.eda import EDA

def test_data_loading():
    # Mock CSV loading with a small DataFrame
    loader = DataLoader()
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    loader.fraud_data = df
    assert loader.fraud_data.equals(df)

def test_data_cleaning():
    cleaner = DataCleaner()
    df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
    cleaned = cleaner.handle_missing_values(df.copy(), critical_fields=['a'], impute_fields={'b': 'median'})
    assert cleaned.isnull().sum().sum() == 0
    deduped = cleaner.remove_duplicates(pd.DataFrame({'a': [1, 1], 'b': [2, 2]}))
    assert len(deduped) == 1
    typed = cleaner.convert_data_types(pd.DataFrame({'a': [1]}), {'a': 'float64'})
    assert typed['a'].dtype == 'float64'

def test_feature_engineering():
    fe = FeatureEngineer()
    df = pd.DataFrame({'time': pd.to_datetime(['2021-01-01 10:00', '2021-01-02 15:00'])})
    df = fe.create_time_features(df, 'time')
    assert 'hour_of_day' in df.columns and 'day_of_week' in df.columns
    df['end'] = pd.to_datetime(['2021-01-01 12:00', '2021-01-02 18:00'])
    diff = fe.calculate_time_difference(df, 'time', 'end', unit='hours')
    assert all(diff == [2, 3])
    df['user'] = [1, 1]
    df = fe.create_transaction_features(df, 'user')
    assert 'transaction_count' in df.columns

def test_eda_summary_stats():
    eda = EDA()
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    stats = eda.generate_summary_stats(df)
    assert 'a' in stats.columns and 'b' in stats.columns 