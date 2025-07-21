import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from imbalance_handling import ImbalanceHandler
from data_split import DataSplitter

def run_logistic_regression(data_path, target_col):
    """
    Run logistic regression on the given dataset and return evaluation metrics.
    Args:
        data_path (str): Path to the CSV file.
        target_col (str): Name of the target column.
    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1, roc_auc, classification_report)
    """
    # Initialize components
    splitter = DataSplitter()
    imbalance_handler = ImbalanceHandler()

    # Load and split data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("\nDataFrame Info:")
    print(df.info())
    print("\nColumns in DataFrame:", df.columns.tolist())

    # Check target column
    if target_col not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == target_col.lower():
                print(f"\nFound target column '{col}' (case-insensitive match for '{target_col}')")
                target_col = col
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns.tolist()}")

    X, y = splitter.separate_features_and_target(df, target_col)
    assert X is not None and y is not None, "Features or target is None after splitting!"

    # Train/test split
    X_train, X_test, y_train, y_test = splitter.train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)
    assert X_train is not None and y_train is not None, "Train split failed!"
    assert X_test is not None and y_test is not None, "Test split failed!"

    # Print class distribution before SMOTE
    print('\nClass distribution in y_train before SMOTE:', y_train.value_counts().to_dict())

    # Keep only numeric columns for SMOTE and modeling
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])
    print('\nColumns used for modeling:', list(X_train_numeric.columns))

    # Handle imbalance ONLY on training data
    X_train_bal, y_train_bal = imbalance_handler.apply_smote(X_train_numeric, y_train)
    assert X_train_bal is not None and y_train_bal is not None, "SMOTE returned None!"
    print('Class distribution in y_train after SMOTE:', pd.Series(y_train_bal).value_counts().to_dict())

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
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
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics

def print_metrics(metrics):
    print('Logistic Regression Baseline Results:')
    print('Accuracy:', metrics['accuracy'])
    print('Precision:', metrics['precision'])
    print('Recall:', metrics['recall'])
    print('F1 Score:', metrics['f1'])
    print('ROC-AUC:', metrics['roc_auc'])
    from sklearn.metrics import classification_report
    # Print the text report
    print('\nClassification Report:\n', classification_report(metrics['classification_report']))

def main():
    # Allow user to specify which dataset to use
    if len(sys.argv) > 1 and sys.argv[1] == 'credit':
        data_path = '../data/processed/credit_minmax_scaled.csv'
        target_col = 'Class'
        print('Using credit_minmax_scaled.csv')
    else:
        data_path = '../data/processed/fraud_one_hot_encoded.csv'
        target_col = 'class'
        print('Using fraud_one_hot_encoded.csv')

    metrics = run_logistic_regression(data_path, target_col)
    print_metrics(metrics)

if __name__ == "__main__":
    main() 