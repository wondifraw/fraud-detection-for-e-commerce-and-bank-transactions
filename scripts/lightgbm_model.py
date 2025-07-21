import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score, precision_recall_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from imbalance_handling import ImbalanceHandler

def run_lightgbm(data_path, target_col):
    """
    Run LightGBM on the given dataset and return evaluation metrics.
    Args:
        data_path (str): Path to the CSV file.
        target_col (str): Name of the target column.
    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1, roc_auc, pr_auc, classification_report)
    """
    imbalance_handler = ImbalanceHandler()

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("\nDataFrame Info:")
    print(df.info())
    print("\nColumns in DataFrame:", df.columns.tolist())

    # Check target column
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

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('\nClass distribution in y_train before SMOTE:', y_train.value_counts().to_dict())

    # Keep only numeric columns for SMOTE and modeling
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])
    print('\nColumns used for modeling:', list(X_train_numeric.columns))

    # Handle imbalance ONLY on training data
    X_train_bal, y_train_bal = imbalance_handler.apply_smote(X_train_numeric, y_train)
    print('Class distribution in y_train after SMOTE:', pd.Series(y_train_bal).value_counts().to_dict())

    # LightGBM with class_weight
    model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_bal, y_train_bal)

    # Predict and evaluate
    y_pred = model.predict(X_test_numeric)
    y_proba = model.predict_proba(X_test_numeric)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {metrics["pr_auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Cross-validation (Stratified)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        lgb.LGBMClassifier(random_state=42, class_weight='balanced'),
        X.select_dtypes(include=['float64', 'int64']), y, cv=skf, scoring='average_precision'
    )
    print(f"\nStratified 5-Fold CV (AUC-PR): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance (LightGBM)
    feature_importance = pd.Series(model.feature_importances_, index=X_train_numeric.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("\nTop 10 Feature Importances (LightGBM):")
    print(feature_importance.head(10))
    feature_importance.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances (LightGBM)')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.show()

    return metrics

def print_metrics(metrics):
    print('LightGBM Results:')
    print('Accuracy:', metrics['accuracy'])
    print('Precision:', metrics['precision'])
    print('Recall:', metrics['recall'])
    print('F1 Score:', metrics['f1'])
    print('ROC-AUC:', metrics['roc_auc'])
    print('PR-AUC:', metrics['pr_auc'])
    from sklearn.metrics import classification_report
    print('\nClassification Report:\n', classification_report(metrics['classification_report']))

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'credit':
        data_path = '../data/processed/credit_minmax_scaled.csv'
        target_col = 'Class'
        print('Using credit_minmax_scaled.csv')
    else:
        data_path = '../data/processed/fraud_one_hot_encoded.csv'
        target_col = 'class'
        print('Using fraud_one_hot_encoded.csv')

    metrics = run_lightgbm(data_path, target_col)
    print_metrics(metrics)

if __name__ == "__main__":
    main() 