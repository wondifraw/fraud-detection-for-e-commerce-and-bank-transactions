import pandas as pd
import numpy as np
import pytest
from src.geolocation import GeolocationProcessor

def test_ip_to_int():
    geo = GeolocationProcessor()
    df = pd.Series(['8.8.8.8', '127.0.0.1'])
    result = geo.ip_to_int(df)
    assert not result.isnull().any()

def test_ip_to_int_invalid():
    geo = GeolocationProcessor()
    df = pd.Series(['invalid_ip', None])
    result = geo.ip_to_int(df)
    assert result.isnull().all()

def test_merge_with_country_data():
    geo = GeolocationProcessor()
    fraud_df = pd.DataFrame({'ip_int': [100, 200]})
    ip_df = pd.DataFrame({'lower_bound_ip_address': [50, 150], 'upper_bound_ip_address': [150, 250], 'country': ['A', 'B']})
    result = geo.merge_with_country_data(fraud_df, ip_df)
    assert 'country' in result.columns
    assert set(result['country']) <= {'A', 'B', 'Unknown'}

def test_find_country():
    geo = GeolocationProcessor()
    ip_df = pd.DataFrame({'lower_bound_ip_address': [0], 'upper_bound_ip_address': [100], 'country': ['A']})
    country = geo._find_country(50, ip_df)
    assert country == 'A'
    country = geo._find_country(200, ip_df)
    assert country == 'Unknown' 