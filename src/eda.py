
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

    def univariate_analysis(self, df, columns=None):
        """Plot distributions for up to 2 columns: prioritize 1 numeric and 1 categorical if available."""
        import numpy as np
        if columns is not None:
            numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
            categorical_cols = [col for col in columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
        else:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        plots_done = 0
        # Plot at most 1 numeric
        if numeric_cols:
            col = numeric_cols[0]
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col].dropna(), bins=30, kde=True)
                plt.title(f'Univariate Distribution: {col}')
                plt.show()
                plots_done += 1
            except Exception as e:
                print(f"Error plotting numeric column {col}: {str(e)}")
        # Plot at most 1 categorical
        if categorical_cols and plots_done < 2:
            col = categorical_cols[0]
            try:
                plt.figure(figsize=(8, 5))
                sns.countplot(y=col, data=df, order=df[col].value_counts().index)
                plt.title(f'Univariate Count Plot: {col}')
                plt.show()
                plots_done += 1
            except Exception as e:
                print(f"Error plotting categorical column {col}: {str(e)}")
        # If only one type exists, plot a second from that type
        if plots_done < 2:
            if len(numeric_cols) > 1:
                col = numeric_cols[1]
                try:
                    plt.figure(figsize=(8, 5))
                    sns.histplot(df[col].dropna(), bins=30, kde=True)
                    plt.title(f'Univariate Distribution: {col}')
                    plt.show()
                    plots_done += 1
                except Exception as e:
                    print(f"Error plotting numeric column {col}: {str(e)}")
            elif len(categorical_cols) > 1:
                col = categorical_cols[1]
                try:
                    plt.figure(figsize=(8, 5))
                    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
                    plt.title(f'Univariate Count Plot: {col}')
                    plt.show()
                    plots_done += 1
                except Exception as e:
                    print(f"Error plotting categorical column {col}: {str(e)}")

    def bivariate_analysis(self, df, columns=None, target_col=None):
        """Plot at most 2 bivariate plots: correlation heatmap and one pairplot or boxplot."""
        import numpy as np
        plots_done = 0
        if columns is not None:
            numeric_cols = [col for col in columns if np.issubdtype(df[col].dtype, np.number)]
        else:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        # Correlation heatmap (always first if possible)
        if len(numeric_cols) > 1:
            try:
                plt.figure(figsize=(12, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=False, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                plt.show()
                plots_done += 1
            except Exception as e:
                print(f"Error plotting correlation heatmap: {str(e)}")
        # Pairplot (scatterplot matrix) or boxplot (if target_col)
        if plots_done < 2:
            if len(numeric_cols) > 1:
                try:
                    sns.pairplot(df[numeric_cols].dropna().iloc[:, :2])
                    plt.suptitle('Pairwise Scatter Plots', y=1.02)
                    plt.show()
                    plots_done += 1
                except Exception as e:
                    print(f"Error plotting pairplot: {str(e)}")
            elif target_col and target_col in df.columns:
                for col in numeric_cols:
                    if col != target_col:
                        try:
                            plt.figure(figsize=(8, 5))
                            sns.boxplot(x=target_col, y=col, data=df)
                            plt.title(f'{col} by {target_col}')
                            plt.show()
                            plots_done += 1
                            break
                        except Exception as e:
                            print(f"Error plotting boxplot for {col} vs {target_col}: {str(e)}")