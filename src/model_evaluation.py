import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, precision_recall_curve, confusion_matrix
)

def evaluate_model(model, X_test, y_test):
    X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])
    y_pred = model.predict(X_test_numeric)
    y_proba = model.predict_proba(X_test_numeric)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics, y_pred, y_proba

def plot_precision_recall(y_test, y_proba, pr_auc, model_name='Model'):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10, model_type='lightgbm'):
    if model_type == 'lightgbm':
        importance = pd.Series(model.feature_importances_, index=feature_names)
    elif model_type == 'logistic':
        importance = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    else:
        raise ValueError('Unknown model_type')
    importance = importance.sort_values(ascending=False)
    print(f"\nTop {top_n} Feature Importances ({model_type.title()}):")
    print(importance.head(top_n))
    importance.head(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Feature Importances ({model_type.title()})')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.show() 