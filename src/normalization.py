
"""
This module provides the DataNormalizer class, which offers methods for normalizing and encoding features
in fraud detection datasets. The main functionalities include:

- Standard scaling: Scales numeric features to have zero mean and unit variance.
- Min-max scaling: Scales numeric features to a specified range, typically [0, 1].
- One-hot encoding: Converts categorical variables into a set of binary columns.

These utilities are designed to prepare data for machine learning models by ensuring features are on comparable scales
and categorical variables are properly encoded.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataNormalizer:
    # The following function applies standard scaling to the specified numeric columns in the dataframe.
    def standard_scale(self, df, columns):
        """Apply standard scaling to specified columns"""
        try:
            # Create an explicit copy of the DataFrame
            df_copy = df.copy()
            scaler = StandardScaler()
            df_copy[columns] = scaler.fit_transform(df_copy[columns])
            return df_copy
        except Exception as e:
            print(f"Error in standard scaling: {str(e)}")
            return None
    # The following function applies min-max scaling to the specified numeric columns in the dataframe.
    def minmax_scale(self, df, columns):
        """Apply min-max scaling to specified columns"""
        try:
            # Create an explicit copy of the DataFrame
            df_copy = df.copy()
            scaler = MinMaxScaler()
            df_copy[columns] = scaler.fit_transform(df_copy[columns])
            return df_copy
        except Exception as e:
            print(f"Error in min-max scaling: {str(e)}")
            return None
    # The following function applies robust scaling to the specified numeric columns in the dataframe.
    def one_hot_encode(self, df, columns):
        """Apply one-hot encoding to categorical columns"""
        try:
            return pd.get_dummies(df, columns=columns)
        except Exception as e:
            print(f"Error in one-hot encoding: {str(e)}")
            return None