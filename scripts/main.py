
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

def main():
    # Initialize all components
    try:
        loader = DataLoader()
        cleaner = DataCleaner()
        eda = EDA()
        geo_processor = GeolocationProcessor()
        feature_engineer = FeatureEngineer()
        imbalance_handler = ImbalanceHandler()
        normalizer = DataNormalizer()
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    # Load data
    try:
        loader.load_fraud_data('../data/raw/Fraud_Data.csv')
        loader.load_ip_data('../data/raw/IpAddress_to_Country.csv')
        loader.load_credit_data('../data/raw/creditcard.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Clean data
    try:
        fraud_data_clean = cleaner.handle_missing_values(
            loader.fraud_data,
            critical_fields=['user_id', 'purchase_value', 'class'],
            impute_fields={'age': 'median', 'sex': 'mode'}
        )
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return
    # --- EDA Step ---
    try:
        eda.generate_summary_stats(fraud_data_clean)
        eda.plot_class_distribution(fraud_data_clean, target_col='class')
        eda.plot_feature_distributions(fraud_data_clean, features=['purchase_value', 'age'])
    except Exception as e:
        print(f"Error during EDA: {e}")
    # --- Geolocation Enrichment ---
    try:
        if 'ip_address' not in fraud_data_clean.columns:
            raise KeyError("'ip_address' column not found in fraud_data_clean. Please check your data and column names.")
        fraud_data_clean['ip_int'] = geo_processor.ip_to_int(fraud_data_clean['ip_address'])
        fraud_data_geo = geo_processor.merge_with_country_data(
            fraud_data_clean,
            loader.ip_data
        )
    except Exception as e:
        print(f"Error during geolocation enrichment: {e}")
        return
    # --- Feature Engineering ---
    try:
        fraud_data_geo['signup_time'] = pd.to_datetime(fraud_data_geo['signup_time'])
        fraud_data_geo['purchase_time'] = pd.to_datetime(fraud_data_geo['purchase_time'])
        fraud_data_geo = feature_engineer.create_time_features(fraud_data_geo, 'signup_time')
        fraud_data_geo = feature_engineer.create_time_features(fraud_data_geo, 'purchase_time')
        fraud_data_geo['signup_to_purchase_hours'] = feature_engineer.calculate_time_difference(
            fraud_data_geo, 'signup_time', 'purchase_time', unit='hours'
        )
        fraud_data_geo = feature_engineer.create_transaction_features(fraud_data_geo, 'user_id')
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return
    # --- Imbalance Handling ---
    try:
        feature_cols = [col for col in fraud_data_geo.columns if col not in ['class', 'ip_address', 'signup_time', 'purchase_time']]
        X = fraud_data_geo[feature_cols]
        y = fraud_data_geo['class']
        X_balanced, y_balanced = imbalance_handler.apply_smote(X, y)
    except Exception as e:
        print(f"Error during imbalance handling: {e}")
        return
    # --- Normalization ---
    try:
        numeric_cols = X_balanced.select_dtypes(include=['float64', 'int64']).columns.tolist()
        X_scaled = normalizer.standard_scale(X_balanced, numeric_cols)
        categorical_cols = X_scaled.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_final = normalizer.one_hot_encode(X_scaled, categorical_cols)
        else:
            X_final = X_scaled
    except Exception as e:
        print(f"Error during normalization: {e}")
        return
    # Continue with other processing steps...
    # (Additional code would follow this pattern)

if __name__ == "__main__":
    main()