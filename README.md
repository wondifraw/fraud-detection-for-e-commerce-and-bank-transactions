# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview

This repository provides a comprehensive, modular, and reproducible pipeline for detecting fraudulent transactions in both e-commerce and banking domains. The project is designed to support robust experimentation, deep data analysis, and production-ready model development.

**Objectives:**
- Detect fraudulent transactions in real-world e-commerce and credit card datasets.
- Provide a flexible, extensible codebase for research and deployment.
- Enable transparent, reproducible, and explainable machine learning workflows.

---

## Task Procedure (Orderly Pipeline)

Below is the step-by-step procedure of the fraud detection pipeline, reflecting the actual workflow order:

1. **Clone the Repository and Set Up Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/wondifraw/fraud-detection-for-e-commerce-and-bank-transactions.git
     cd fraud-detection-for-e-commerce-and-bank-transactions
     ```
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     # On Unix/macOS:
     source venv/bin/activate
     # On Windows:
     venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Understand the Repository Structure**
   - **data/**: Raw and processed datasets.
   - **notebooks/**: Jupyter notebooks for EDA, preprocessing, modeling, and evaluation.
   - **src/**: Modular Python scripts for each pipeline stage.
   - **tests/**: Unit tests for reliability.
   - **requirements.txt**: Pinned dependencies.
   - **README.md**: Documentation and usage.

3. **Data Loading**
   - Import raw datasets (e-commerce, credit card, IP geolocation) using the `DataLoader` module.

4. **Data Cleaning**
   - Handle missing values, remove duplicates, and enforce data types with `DataCleaner`.

5. **Exploratory Data Analysis (EDA)**
   - Generate summary statistics and visualizations using the EDA module or Jupyter notebooks.

6. **Feature Engineering**
   - Create time-based, frequency, and aggregation features with `FeatureEngineer`.

7. **Geolocation Enrichment**
   - Map IP addresses to countries for risk analysis using `GeolocationProcessor`.

8. **Imbalance Handling**
   - Apply SMOTE oversampling and random undersampling with `ImbalanceHandler`.

9. **Normalization & Encoding**
   - Scale numeric features and encode categoricals using `DataNormalizer`.

10. **Data Splitting**
    - Perform stratified train/test split to preserve class balance.

11. **Model Training**
    - Train Logistic Regression and LightGBM models using the modeling scripts.

12. **Evaluation**
    - Assess models with comprehensive metrics and visualizations (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices).

13. **Model Explainability (SHAP)**
    - Use the SHAP module to interpret model predictions and feature importance.

14. **Reproducibility and Testing**
    - Run unit tests in `tests/`:
      ```bash
      pytest tests/
      ```
    - All experiments are reproducible with pinned dependencies.

15. **Extending the Project**
    - Add new models or feature engineering steps by extending scripts in `src/`.
    - Integrate additional datasets by placing them in `data/raw/` and updating data loading scripts.
    - Use provided notebooks as templates for further analysis or reporting.
    - Customize the pipeline by modifying or adding steps in `scripts/main.py`.

---

## Pipeline Diagram
