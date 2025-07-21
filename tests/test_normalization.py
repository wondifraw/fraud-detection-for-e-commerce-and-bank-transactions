import pandas as pd
import pytest
from src.normalization import DataNormalizer

def test_standard_scale():
    norm = DataNormalizer()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = norm.standard_scale(df, ['a'])
    assert abs(result['a'].mean()) < 1e-6
    assert abs(result['a'].std() - 1) < 1e-6

def test_standard_scale_empty():
    norm = DataNormalizer()
    df = pd.DataFrame({'a': []})
    result = norm.standard_scale(df, ['a'])
    assert result is not None

def test_minmax_scale():
    norm = DataNormalizer()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = norm.minmax_scale(df, ['a'])
    assert result['a'].min() == 0 and result['a'].max() == 1

def test_minmax_scale_empty():
    norm = DataNormalizer()
    df = pd.DataFrame({'a': []})
    result = norm.minmax_scale(df, ['a'])
    assert result is not None

def test_one_hot_encode():
    norm = DataNormalizer()
    df = pd.DataFrame({'cat': ['x', 'y', 'x']})
    result = norm.one_hot_encode(df, ['cat'])
    assert 'cat_x' in result.columns and 'cat_y' in result.columns

def test_one_hot_encode_empty():
    norm = DataNormalizer()
    df = pd.DataFrame({'cat': []})
    result = norm.one_hot_encode(df, ['cat'])
    assert result is not None 