
"""
This module provides the ImbalanceHandler class, which offers methods for handling class imbalance
in fraud detection datasets. The main functionalities include:

- SMOTE oversampling: Generates synthetic samples for the minority class to balance the dataset.
- Random undersampling: Reduces the size of the majority class to match the minority class.

These utilities are designed to help improve model performance by addressing class imbalance issues
commonly encountered in fraud detection workflows.
"""


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

class ImbalanceHandler:
    # The following function applies SMOTE oversampling to balance the classes in the feature matrix X and target vector y.
    def apply_smote(self, X, y):
        """Apply SMOTE oversampling"""
        try:
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X, y)
        except Exception as e:
            print(f"Error applying SMOTE: {str(e)}")
            return None, None
    # The following function applies random undersampling to balance the classes in the dataframe by reducing the majority class.
    def random_undersample(self, df, target_col):
        """Apply random undersampling"""
        try:
            # Separate majority and minority classes
            majority = df[df[target_col] == 0]
            minority = df[df[target_col] == 1]
            
            # Downsample majority class
            majority_downsampled = resample(
                majority,
                replace=False,
                n_samples=len(minority),
                random_state=42
            )
            
            # Combine with minority class
            return pd.concat([majority_downsampled, minority])
        except Exception as e:
            print(f"Error in random undersampling: {str(e)}")
            return None