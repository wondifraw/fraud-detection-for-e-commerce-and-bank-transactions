# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview

This repository delivers a robust, modular, and fully reproducible pipeline for detecting fraudulent transactions in both e-commerce and banking environments. The project is engineered for rigorous experimentation, in-depth data analysis, and seamless transition from research to production deployment.

**Key Objectives:**
- Accurately detect fraudulent transactions in real-world e-commerce and credit card datasets.
- Provide a flexible, extensible, and well-documented codebase for both research and operational use.
- Ensure transparency, reproducibility, and explainability throughout the machine learning workflow.

---

## Repository Structure

- **data/**: Contains raw, intermediate, and processed datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA), preprocessing, modeling, and evaluation.
- **src/**: Modular Python scripts for each pipeline stage (data loading, cleaning, feature engineering, modeling, etc.).
- **scripts/**: Main pipeline orchestration scripts (e.g., `main.py`).
- **tests/**: Unit tests to ensure reliability and correctness of each module.
- **requirements.txt**: Pinned Python dependencies for full reproducibility.
- **README.md**: Comprehensive documentation and usage instructions.

---

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/wondifraw/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```

2. **Set Up a Python Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Jupyter Notebook for Interactive Analysis**
   ```bash
   pip install notebook
   ```

5. **Verify Installation**
   - Run the unit tests to ensure all modules are working:
     ```bash
     pytest tests/
     ```

6. **Download or Place Data**
   - Place your raw datasets in the `data/raw/` directory as described in the project documentation.
---

## End-to-End Pipeline: Step-by-Step Procedure

The following steps reflect the actual workflow implemented in the codebase, ensuring comprehensive and correct execution of all technical tasks:

1. **Data Loading**
   - Use the `DataLoader` module to import raw datasets, including e-commerce transactions, credit card records, and IP geolocation mappings.

2. **Rigorous Data Cleaning**
   - Employ `DataCleaner` to:
     - Identify and handle missing values using appropriate imputation or removal strategies.
     - Remove duplicate records.
     - Enforce correct data types and resolve inconsistencies.
     - Validate data integrity before further processing.

3. **Exploratory Data Analysis (EDA)**
   - Generate descriptive statistics, visualize distributions, and identify patterns or anomalies using the EDA module or provided Jupyter notebooks.

4. **Comprehensive Feature Engineering**
   - Apply `FeatureEngineer` to create and document:
     - Time-based features (e.g., transaction hour, day of week).
     - Frequency and aggregation features (e.g., transaction counts per user/IP).
     - Domain-specific features relevant to fraud detection.

5. **Geolocation Enrichment**
   - Use `GeolocationProcessor` to map IP addresses to countries or regions, enabling geospatial risk analysis.

6. **Imbalance Handling**
   - Address class imbalance with `ImbalanceHandler`:
     - Apply SMOTE (Synthetic Minority Over-sampling Technique) for minority class augmentation.
     - Optionally combine with random undersampling of the majority class.

7. **Preprocessing: Normalization & Encoding**
   - Use `DataNormalizer` to:
     - Scale numeric features (e.g., MinMaxScaler, StandardScaler).
     - Encode categorical variables (e.g., one-hot encoding, label encoding) as required for modeling.

8. **Data Splitting**
   - Perform stratified train/test splits to preserve class distribution and ensure fair model evaluation.

9. **Model Training**
   - Train multiple models using modular scripts:
     - **Logistic Regression**: For interpretable baseline performance.
     - **LightGBM**: For high-performance gradient boosting.
   - Hyperparameters and feature selection are documented and can be tuned as needed.

10. **Comprehensive Evaluation**
    - Evaluate models using a suite of metrics:
      - Accuracy, Precision, Recall, F1-score
      - ROC-AUC, PR-AUC
      - Confusion matrices
    - Visualize results with precision-recall curves and feature importance plots.

11. **Model Explainability**
    - Leverage SHAP (SHapley Additive exPlanations) to interpret model predictions and understand feature contributions.

12. **Reproducibility and Testing**
    - All experiments are reproducible with pinned dependencies in `requirements.txt`.
    - Run unit tests to verify correctness:
      ```bash
      pytest tests/
      ```

13. **Extending the Project**
    - Add new models, feature engineering steps, or data sources by extending scripts in `src/`.
    - Integrate additional datasets by placing them in `data/raw/` and updating data loading scripts.
    - Use and adapt provided notebooks for further analysis or reporting.
    - Customize the pipeline by modifying or adding steps in `scripts/main.py`.

---

## Usage Example

To run the end-to-end fraud detection pipeline, execute the main script from the `scripts/` directory. You can specify which dataset to use via a command-line argument:

### Running the Pipeline in Jupyter Notebooks

You can also interactively run and experiment with the pipeline using Jupyter notebooks. This is useful for step-by-step exploration, visualization, and custom analysis.

1. **Start Jupyter Notebook**
   From the project root or `scripts/` directory, launch Jupyter:

   ```bash
   jupyter notebook
   ```

2. **Import Pipeline Modules**  
   In a notebook cell, you can import and use the pipeline components directly. For example:

   ```python
   import sys
   import os
   sys.path.append(os.path.abspath('..'))  # Ensure src/ is in the path

   import pandas as pd
   from src.data_cleaning import DataCleaner
   from src.eda import EDA
   from src.feature_engineering import FeatureEngineer
   from src.geolocation import GeolocationProcessor
   from src.normalization import DataNormalizer
   from src.model_training import prepare_data, train_logistic_regression, train_lightgbm
   from src.model_evaluation import evaluate_model
   ```

3. **Step Through the Pipeline**  
   You can manually execute each step (data loading, cleaning, EDA, feature engineering, modeling, etc.) in separate cells for full control and visualization.

4. **Customize and Visualize**  
   Use notebook cells to:
   - Visualize data distributions and model results with `matplotlib` or `seaborn`.
   - Experiment with different preprocessing or modeling strategies.
   - Document findings and create reports.

5. **Example Notebook**  
   See the provided example notebook(s) in the `notebooks/` directory for a guided walkthrough of the pipeline.

---
## Results

After running the end-to-end fraud detection pipeline, here are some sample results and key findings from the baseline models:

### Model Performance

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|----------------------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression  | 0.98     | 0.85      | 0.76   | 0.80     | 0.97    | 0.72   |
| LightGBM             | 0.99     | 0.91      | 0.81   | 0.86     | 0.99    | 0.80   |

*Note: These are representative results; actual values may vary depending on the dataset and random seed.*

### Key Findings

- **Class Imbalance**: The dataset is highly imbalanced, with fraudulent transactions making up less than 1% of all records. Handling imbalance (e.g., with SMOTE) is crucial for meaningful model performance.
- **Feature Importance**: Both models identified transaction amount, time-based features, and geolocation-derived features as highly predictive for fraud detection.
- **Model Comparison**: LightGBM outperformed Logistic Regression across most metrics, especially in recall and PR-AUC, indicating better detection of rare fraud cases.
- **Explainability**: SHAP analysis highlighted that unusually high transaction amounts and mismatches between IP geolocation and cardholder country are strong fraud indicators.

### Sharp Analysis Output
Below is an example of SHAP summary output for the LightGBM model, visualizing the most important features influencing fraud predictions:

*Interpretation*:  
- Features at the top (e.g., `transaction_amount`, `ip_country_mismatch`, `transaction_hour`) have the greatest impact on the model's output.
- Red points indicate higher feature values, blue points indicate lower values.
- For example, high `transaction_amount` and a mismatch between IP and cardholder country strongly increase the likelihood of a transaction being classified as fraud.

> *To reproduce this plot, run the pipeline and ensure SHAP is installed. The plot will be saved in the `figures/` directory if you enable saving in your script or notebook.*


---
## Future Work

There are several directions for extending and improving this fraud detection pipeline:

- **Advanced Feature Engineering**: Incorporate additional domain-specific features, interaction terms, or automated feature selection techniques.
- **Model Ensembling**: Combine multiple models (e.g., stacking, bagging, boosting) to improve predictive performance and robustness.
- **Hyperparameter Optimization**: Integrate automated hyperparameter tuning (e.g., with `GridSearchCV`, `RandomizedSearchCV`, or `Optuna`) for better model selection.
- **Additional Algorithms**: Experiment with other machine learning algorithms such as XGBoost, CatBoost, or neural networks.
- **Explainability and Interpretability**: Expand model explainability using SHAP, LIME, or other interpretability tools for deeper insights.
- **Real-Time Scoring**: Adapt the pipeline for real-time or batch scoring in production environments.
- **Data Drift and Monitoring**: Add tools for monitoring data drift, model performance over time, and automated retraining triggers.
- **Automated Reporting**: Generate automated reports or dashboards summarizing results and key metrics.
- **Integration with External Data**: Enrich datasets with additional external sources (e.g., device fingerprinting, behavioral analytics).
- **Robust Testing and CI/CD**: Expand unit/integration tests and set up continuous integration for reliable deployments.












