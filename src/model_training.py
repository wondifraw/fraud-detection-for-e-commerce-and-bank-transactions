import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from .imbalance_handling import ImbalanceHandler
from .data_split import separate_features_and_target, stratified_train_test_split

def prepare_data(df, target_col, test_size=0.2, random_state=42):
    X, y = separate_features_and_target(df, target_col)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    imbalance_handler = ImbalanceHandler()
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_train_bal, y_train_bal = imbalance_handler.apply_smote(X_train_numeric, y_train)
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_bal, y_train_bal)
    return model, X_train_bal.columns

def train_lightgbm(X_train, y_train):
    imbalance_handler = ImbalanceHandler()
    X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
    X_train_bal, y_train_bal = imbalance_handler.apply_smote(X_train_numeric, y_train)
    model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_bal, y_train_bal)
    return model, X_train_bal.columns

def cross_validate_model(model, X, y, scoring='average_precision', n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_numeric = X.select_dtypes(include=['float64', 'int64'])
    cv_scores = cross_val_score(model, X_numeric, y, cv=skf, scoring=scoring)
    return cv_scores 