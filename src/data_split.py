import pandas as pd
from sklearn.model_selection import train_test_split

def separate_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Separates features and target from a DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    if target_col not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == target_col.lower():
                target_col = col
                break
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train-test split.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test set.
        random_state (int): Random seed.
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 