
"""
This module provides the FeatureEngineer class, which offers methods for creating new features
from raw data to enhance fraud detection models. The main functionalities include:

- Creating time-based features: Extracts components such as hour of day and day of week from timestamps.
- Calculating time differences: Computes the time elapsed between two datetime columns in various units.
- Generating transaction frequency features: Counts the number of transactions per unique identifier.

These utilities are designed to enrich datasets with informative features for improved model performance.
"""


import pandas as pd

class FeatureEngineer:
    # The following function creates time-based features (hour of day, day of week) from a timestamp column.
    def create_time_features(self, df, time_col):
        """Create time-based features from timestamp"""
        try:
            df['hour_of_day'] = df[time_col].dt.hour
            df['day_of_week'] = df[time_col].dt.dayofweek
            return df
        except Exception as e:
            print(f"Error creating time features: {str(e)}")
            return None
    # The following function calculates the time difference between two datetime columns in the dataframe.
    def calculate_time_difference(self, df, start_col, end_col, unit='hours'):
        """Calculate time difference between two columns"""
        try:
            time_diff = (df[end_col] - df[start_col]).dt.total_seconds()
            if unit == 'hours':
                time_diff = time_diff / 3600
            elif unit == 'days':
                time_diff = time_diff / (3600 * 24)
            return time_diff
        except Exception as e:
            print(f"Error calculating time difference: {str(e)}")
            return None
    # The following function creates transaction frequency features by counting the number of transactions per unique identifier.
    def create_transaction_features(self, df, id_col):
        """Create transaction frequency features"""
        try:
            trans_counts = df[id_col].value_counts().to_dict()
            df['transaction_count'] = df[id_col].map(trans_counts)
            return df
        except Exception as e:
            print(f"Error creating transaction features: {str(e)}")
            return None