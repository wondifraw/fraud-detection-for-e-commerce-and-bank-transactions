import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def separate_features_and_target(df, target_col):
    """
    Splits a DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets using stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
