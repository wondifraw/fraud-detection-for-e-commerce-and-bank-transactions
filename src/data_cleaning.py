
"""
This module provides the DataCleaner class, which offers methods for cleaning and preprocessing
datasets used in fraud detection workflows. The main functionalities include:

- Handling missing values: Drops rows with missing critical fields and imputes other fields using
  specified strategies (e.g., median, mode).
- Removing duplicate rows: Ensures data integrity by eliminating duplicate records.
- Converting data types: Converts columns to specified data types for consistency and analysis.

These utilities are designed to prepare raw data for further analysis and modeling.
"""


import pandas as pd
# The DataCleaner class provides methods for cleaning and preprocessing dataframes,
# including handling missing values, removing duplicates, and converting data types.

class DataCleaner:
    # The following function handles missing values in the dataframe by dropping rows with missing critical fields
    # and imputing other specified fields using the provided strategies (e.g., median, mode).
    def handle_missing_values(self, df, critical_fields, impute_fields):
        """Handle missing values in dataframe"""
        try:
            # Drop rows with missing critical fields
            df.dropna(subset=critical_fields, inplace=True)
            
            # Impute other fields
            for field, strategy in impute_fields.items():
                if strategy == 'median':
                    df[field] = df[field].fillna(df[field].median())
                elif strategy == 'mode':
                    df[field] = df[field].fillna(df[field].mode()[0])
            return df
        except Exception as e:
            print(f"Error handling missing values: {str(e)}")
            return None
    # The following function removes outliers from specified columns using the IQR method.
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        try:
            return df.drop_duplicates()
        except Exception as e:
            print(f"Error removing duplicates: {str(e)}")
            return None
    # The following function converts columns in the dataframe to specified data types based on a provided mapping.
    def convert_data_types(self, df, type_mapping):
        """Convert data types according to mapping"""
        try:
            for col, dtype in type_mapping.items():
                df.loc[:, col] = df[col].astype(dtype)
            return df
        except Exception as e:
            print(f"Error converting data types: {str(e)}")
            return None