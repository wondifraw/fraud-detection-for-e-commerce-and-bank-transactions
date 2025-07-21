
"""
Main script for orchestrating the fraud detection data pipeline.

This script performs the following steps:
- Loads raw fraud, IP-to-country, and credit card datasets.
- Cleans the loaded data, handling missing values and imputing as needed.
- Initializes components for EDA, geolocation enrichment, feature engineering, class imbalance handling, and normalization.
- Serves as the entry point for the end-to-end data processing workflow.

Additional processing steps (EDA, feature engineering, etc.) should be added following the provided pattern.
"""
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


from data_loading import DataLoader
from data_cleaning import DataCleaner
from eda import EDA
from geolocation import GeolocationProcessor
from feature_engineering import FeatureEngineer
from imbalance_handling import ImbalanceHandler
from normalization import DataNormalizer

def separate_and_split(data_path, target_col):
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("\nColumns in DataFrame:", df.columns.tolist())
    if target_col not in df.columns:
        for col in df.columns:
            if col.lower() == target_col.lower():
                print(f"\nFound target column '{col}' (case-insensitive match for '{target_col}')")
                target_col = col
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'credit':
        data_path = '../data/processed/credit_minmax_scaled.csv'
        target_col = 'Class'
        print('Using credit_minmax_scaled.csv')
    else:
        data_path = '../data/processed/fraud_one_hot_encoded.csv'
        target_col = 'class'
        print('Using fraud_one_hot_encoded.csv')
    separate_and_split(data_path, target_col)

if __name__ == "__main__":
    main()