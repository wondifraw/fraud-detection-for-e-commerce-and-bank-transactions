
"""
This module provides the EDA (Exploratory Data Analysis) class, which offers methods for
generating summary statistics and visualizations for datasets used in fraud detection workflows.

Main functionalities include:
- Generating descriptive statistics for all columns in a DataFrame.
- Plotting the distribution of the target variable (e.g., fraud vs. non-fraud).
- Plotting the distribution of numeric columns to understand their spread and detect anomalies.

These utilities are designed to help analysts and data scientists quickly understand the structure,
distribution, and potential issues in their data before proceeding to modeling or further analysis.
"""


import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    # The following function generates descriptive statistics for all columns in the dataframe.
    def generate_summary_stats(self, df):
        """Generate descriptive statistics"""
        try:
            return df.describe(include='all')
        except Exception as e:
            print(f"Error generating summary stats: {str(e)}")
            return None
    # The following function plots the correlation heatmap for numeric features in the dataframe.
    def plot_class_distribution(self, df, target_col):
        """Plot distribution of target variable"""
        try:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=target_col, data=df)
            plt.title('Class Distribution')
            plt.show()
        except Exception as e:
            print(f"Error plotting class distribution: {str(e)}")
    # The following function plots a correlation heatmap for numeric features in the dataframe.
    def plot_numeric_distribution(self, df, col):
        """Plot distribution of numeric column"""
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f'{col} Distribution')
            plt.show()
        except Exception as e:
            print(f"Error plotting {col} distribution: {str(e)}")

    def univariate_analysis(self, df):
        """Plot distributions for all numeric and categorical columns."""
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in numeric_cols:
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col].dropna(), bins=30, kde=True)
                plt.title(f'Univariate Distribution: {col}')
                plt.show()
            except Exception as e:
                print(f"Error plotting numeric column {col}: {str(e)}")
        for col in categorical_cols:
            try:
                plt.figure(figsize=(8, 5))
                sns.countplot(y=col, data=df, order=df[col].value_counts().index)
                plt.title(f'Univariate Count Plot: {col}')
                plt.show()
            except Exception as e:
                print(f"Error plotting categorical column {col}: {str(e)}")

    def bivariate_analysis(self, df, target_col=None):
        """Plot pairwise relationships and correlation heatmap. If target_col is provided, plot boxplots for categorical vs. numeric."""
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Correlation heatmap
        try:
            plt.figure(figsize=(12, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.show()
        except Exception as e:
            print(f"Error plotting correlation heatmap: {str(e)}")
        # Pairplot (scatterplot matrix)
        try:
            if len(numeric_cols) > 1:
                sns.pairplot(df[numeric_cols].dropna())
                plt.suptitle('Pairwise Scatter Plots', y=1.02)
                plt.show()
        except Exception as e:
            print(f"Error plotting pairplot: {str(e)}")
        # Boxplots for categorical vs. numeric (if target_col is provided)
        if target_col and target_col in df.columns:
            for col in numeric_cols:
                if col != target_col:
                    try:
                        plt.figure(figsize=(8, 5))
                        sns.boxplot(x=target_col, y=col, data=df)
                        plt.title(f'{col} by {target_col}')
                        plt.show()
                    except Exception as e:
                        print(f"Error plotting boxplot for {col} vs {target_col}: {str(e)}")