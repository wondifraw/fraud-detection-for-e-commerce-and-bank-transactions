
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

# Ensure src/ is in the path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
from data_loading import DataLoader
from data_cleaning import DataCleaner
from eda import EDA
from geolocation import GeolocationProcessor
from feature_engineering import FeatureEngineer
from imbalance_handling import ImbalanceHandler
from normalization import DataNormalizer
from model_training import prepare_data, train_logistic_regression, train_lightgbm, cross_validate_model
from model_evaluation import evaluate_model, plot_precision_recall, plot_feature_importance

def main():
    # Choose dataset based on command-line argument
    if len(sys.argv) > 1 and sys.argv[1] == 'credit':
        data_path = '../data/processed/credit_minmax_scaled.csv'
        target_col = 'Class'
        print('Using credit_minmax_scaled.csv')
    else:
        data_path = '../data/processed/fraud_one_hot_encoded.csv'
        target_col = 'class'
        print('Using fraud_one_hot_encoded.csv')

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded data from: {data_path}")
    print("Columns:", df.columns.tolist())

    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    print("Data cleaned.")

    # EDA (optional, can be expanded)
    eda = EDA()
    eda.basic_stats(df_clean)
    # eda.plot_distributions(df_clean, target_col)  # Uncomment if implemented

    # Feature engineering
    fe = FeatureEngineer()
    df_fe = fe.transform(df_clean)
    print("Feature engineering complete.")

    # Geolocation enrichment (if needed)
    if 'ip_address' in df_fe.columns:
        geo = GeolocationProcessor()
        df_fe = geo.enrich(df_fe)
        print("Geolocation enrichment complete.")

    # Normalization
    normalizer = DataNormalizer()
    df_norm = normalizer.fit_transform(df_fe)
    print("Normalization complete.")

    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data(df_norm, target_col)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")

    # Train models
    print("\nTraining Logistic Regression...")
    logreg_model, logreg_features = train_logistic_regression(X_train, y_train)
    print("Training LightGBM...")
    lgbm_model, lgbm_features = train_lightgbm(X_train, y_train)

    # Evaluate models
    print("\nEvaluating Logistic Regression...")
    logreg_metrics, logreg_pred, logreg_proba = evaluate_model(logreg_model, X_test, y_test)
    print("Logistic Regression metrics:", {k: v for k, v in logreg_metrics.items() if k != 'classification_report' and k != 'confusion_matrix'})
    plot_precision_recall(y_test, logreg_proba, logreg_metrics['pr_auc'], model_name='Logistic Regression')
    plot_feature_importance(logreg_model, logreg_features, model_type='logistic')

    print("\nEvaluating LightGBM...")
    lgbm_metrics, lgbm_pred, lgbm_proba = evaluate_model(lgbm_model, X_test, y_test)
    print("LightGBM metrics:", {k: v for k, v in lgbm_metrics.items() if k != 'classification_report' and k != 'confusion_matrix'})
    plot_precision_recall(y_test, lgbm_proba, lgbm_metrics['pr_auc'], model_name='LightGBM')
    plot_feature_importance(lgbm_model, lgbm_features, model_type='lightgbm')

    # Cross-validation scores
    print("\nCross-validating Logistic Regression...")
    logreg_cv_scores = cross_validate_model(logreg_model, X_train, y_train)
    print("Logistic Regression CV average_precision scores:", logreg_cv_scores)

    print("\nCross-validating LightGBM...")
    lgbm_cv_scores = cross_validate_model(lgbm_model, X_train, y_train)
    print("LightGBM CV average_precision scores:", lgbm_cv_scores)

if __name__ == "__main__":
    main()