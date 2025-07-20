
"""
This module provides the DataLoader class, which is responsible for loading and managing datasets
used in fraud detection workflows. The supported datasets include:

- E-commerce transaction data: Contains records of online transactions, potentially including features
  such as transaction amount, user information, and fraud labels.
- IP-to-country mapping data: Maps IP address ranges to their corresponding countries, useful for
  geolocation and risk analysis.
- Credit card transaction data: Includes details of credit card transactions, which may be used for
  additional fraud analysis or cross-referencing.

The DataLoader class offers methods to load each dataset from CSV files, handling errors gracefully
and storing the loaded data as pandas DataFrames for further processing.
"""


import pandas as pd

class DataLoader:
    def __init__(self):
        self.fraud_data = None
        self.ip_data = None
        self.credit_data = None
    # The following function loads e-commerce transaction data from a CSV file into a DataFrame.
    def load_fraud_data(self, path):
        """Load e-commerce transaction data"""
        try:
            self.fraud_data = pd.read_csv(path)
            return self.fraud_data
        except Exception as e:
            print(f"Error loading fraud data: {str(e)}")
            return None
    # The following function loads IP-to-country mapping data from a CSV file into a DataFrame.
    def load_ip_data(self, path):
        """Load IP to country mapping data"""
        try:
            self.ip_data = pd.read_csv(path)
            return self.ip_data
        except Exception as e:
            print(f"Error loading IP data: {str(e)}")
            return None
    # The following function loads credit card transaction data from a CSV file into a DataFrame.
    def load_credit_data(self, path):
        """Load credit card transaction data"""
        try:
            self.credit_data = pd.read_csv(path)
            return self.credit_data
        except Exception as e:
            print(f"Error loading credit data: {str(e)}")
            return None