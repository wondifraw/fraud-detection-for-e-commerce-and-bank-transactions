"""
SHAP Explainability Script for Fraud Detection Models (fraud & Credit Card data)

This script loads trained LightGBM and Logistic Regression models for both e-commerce fraud and credit card datasets,
computes SHAP values, and generates summary and force plots for both global and local interpretability.
"""
import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

def load_data(data_dir, data_file, target_col):
    """
    Load processed data and split into features and target.
    Args:
        data_dir (str): Directory containing the data file.
        data_file (str): Filename of the processed data.
        target_col (str): Name of the target column.
    Returns:
        X_numeric (pd.DataFrame): Numeric feature columns.
        y (pd.Series): Target labels.
    """
    try:
        df = pd.read_csv(os.path.join(data_dir, data_file))
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Only use numeric columns for modeling (as in training)
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        return X_numeric, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_model(model_dir, model_file):
    """
    Load a trained model from disk using joblib.
    Args:
        model_dir (str): Directory containing the model file.
        model_file (str): Filename of the model.
    Returns:
        model: Loaded model object.
    """
    try:
        model = joblib.load(os.path.join(model_dir, model_file))
        return model
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        raise

def compute_shap_values(model, X, model_type):
    """
    Compute SHAP values for a given model and data.
    Args:
        model: Trained model object.
        X (pd.DataFrame): Feature data.
        model_type (str): 'lightgbm' or 'logistic'.
    Returns:
        explainer: SHAP explainer object.
        shap_values: Computed SHAP values.
    """
    try:
        if model_type == 'lightgbm':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif model_type == 'logistic':
            explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        return explainer, shap_values
    except Exception as e:
        print(f"Error computing SHAP values for {model_type}: {e}")
        raise

def plot_shap_summary(shap_values, X, model_name, out_dir, dataset_name, is_lgbm_binary=False):
    """
    Generate and save SHAP summary (bar and beeswarm) plots.
    Args:
        shap_values: SHAP values array or list.
        X (pd.DataFrame): Feature data.
        model_name (str): Name of the model for plot titles.
        out_dir (str): Directory to save plots.
        dataset_name (str): Name of the dataset for file naming.
        is_lgbm_binary (bool): If True, use shap_values[1] for binary LightGBM.
    """
    try:
        values = shap_values[1] if is_lgbm_binary and isinstance(shap_values, list) else shap_values
        # Bar plot (global feature importance)
        shap.summary_plot(values, X, plot_type='bar', show=False)
        plt.title(f'{model_name} SHAP Feature Importance (Bar) - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_{dataset_name.lower()}_{model_name.lower()}_summary_bar.png'))
        plt.show()
        # Beeswarm plot
        shap.summary_plot(values, X, show=False)
        plt.title(f'{model_name} SHAP Summary (Beeswarm) - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_{dataset_name.lower()}_{model_name.lower()}_summary_beeswarm.png'))
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP summary for {model_name} ({dataset_name}): {e}")

def plot_shap_force(explainer, shap_values, X, model_name, out_dir, dataset_name, sample_idx=0, is_lgbm_binary=False):
    """
    Generate and save a SHAP force plot for a single prediction.
    Args:
        explainer: SHAP explainer object.
        shap_values: SHAP values array or list.
        X (pd.DataFrame): Feature data.
        model_name (str): Name of the model for plot titles.
        out_dir (str): Directory to save plots.
        dataset_name (str): Name of the dataset for file naming.
        sample_idx (int): Index of the sample to explain.
        is_lgbm_binary (bool): If True, use shap_values[1] and explainer.expected_value[1].
    """
    try:
        shap.initjs()
        if is_lgbm_binary and isinstance(shap_values, list):
            values = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            values = shap_values
            expected_value = explainer.expected_value
        shap.force_plot(
            expected_value,
            values[sample_idx, :],
            X.iloc[sample_idx, :],
            matplotlib=True
        )
        plt.title(f'{model_name} SHAP Force Plot (Sample {sample_idx}) - {dataset_name}')
        plt.savefig(os.path.join(out_dir, f'shap_{dataset_name.lower()}_{model_name.lower()}_force_sample{sample_idx}.png'))
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP force plot for {model_name} ({dataset_name}): {e}")

def run_shap_for_dataset(config):
    """
    Run SHAP analysis for a single dataset/model configuration.
    Args:
        config (dict): Configuration dictionary for dataset and models.
    """
    print(f"\n--- SHAP Analysis for {config['dataset_name']} ---")
    X_numeric, y = load_data(config['data_dir'], config['data_file'], config['target_col'])
    # LightGBM
    try:
        lgbm_model = load_model(config['model_dir'], config['lgbm_model_file'])
        explainer_lgbm, shap_values_lgbm = compute_shap_values(lgbm_model, X_numeric, 'lightgbm')
        plot_shap_summary(shap_values_lgbm, X_numeric, 'LightGBM', config['model_dir'], config['dataset_name'], is_lgbm_binary=True)
        plot_shap_force(explainer_lgbm, shap_values_lgbm, X_numeric, 'LightGBM', config['model_dir'], config['dataset_name'], sample_idx=0, is_lgbm_binary=True)
    except Exception as e:
        print(f"LightGBM SHAP analysis failed for {config['dataset_name']}: {e}")
    # Logistic Regression
    try:
        logreg_model = load_model(config['model_dir'], config['logreg_model_file'])
        explainer_logreg, shap_values_logreg = compute_shap_values(logreg_model, X_numeric, 'logistic')
        plot_shap_summary(shap_values_logreg, X_numeric, 'LogReg', config['model_dir'], config['dataset_name'])
        plot_shap_force(explainer_logreg, shap_values_logreg, X_numeric, 'LogReg', config['model_dir'], config['dataset_name'], sample_idx=0)
    except Exception as e:
        print(f"Logistic Regression SHAP analysis failed for {config['dataset_name']}: {e}")

def run_all_shap_analyses():
    """
    Run SHAP explainability for both datasets and both models. Callable from notebooks or scripts.
    """
    # Configuration for both datasets
    base_dir = os.path.dirname(__file__)
    configs = [
        {
            'dataset_name': 'Ecommerce',
            'data_dir': os.path.join(base_dir, '../data/processed'),
            'data_file': 'fraud_one_hot_encoded.csv',
            'target_col': 'class',
            'model_dir': os.path.join(base_dir, '../models'),
            'lgbm_model_file': 'lgbm_fraud_model.joblib',
            'logreg_model_file': 'log_reg_fraud_model.joblib',
        },
        {
            'dataset_name': 'CreditCard',
            'data_dir': os.path.join(base_dir, '../data/processed'),
            'data_file': 'credit_minmax_scaled.csv',
            'target_col': 'Class',
            'model_dir': os.path.join(base_dir, '../models'),
            'lgbm_model_file': 'lgbm_credit_model.joblib',
            'logreg_model_file': 'log_reg_credit_model.joblib',
        }
    ]
    for config in configs:
        run_shap_for_dataset(config)
    print('\nSHAP analysis complete for all datasets. Plots saved in the models directory.') 