# Fraud Detection for E-Commerce and Bank Transactions

## Overview

Fraudulent transactions are a major challenge for both e-commerce platforms and financial institutions, leading to significant financial losses and eroding customer trust. This project builds a robust, end-to-end pipeline for detecting fraudulent activities in transactional data. By leveraging advanced data preprocessing, feature engineering, and state-of-the-art machine learning models, this solution helps organizations identify and prevent fraud in real time, reducing risk and improving operational efficiency.

**Key Features:**
- Modular Python codebase for easy extension and maintenance
- Comprehensive EDA with both univariate and bivariate analysis
- Automated handling of missing data, outliers, and class imbalance
- Geolocation enrichment using IP-to-country mapping
- Feature engineering for time, frequency, and user behavior
- Implementation of Logistic Regression and LightGBM models
- Jupyter notebooks for interactive exploration and reproducibility
- Unit tests for core pipeline components
- Clear separation of features and target, and stratified train-test split via `src/data_split.py`

## Pipeline Description

The pipeline is modular and extensible, covering all critical stages of a modern fraud detection workflow:

1. **Data Loading:**
   - Loads e-commerce, credit card, and IP-to-country datasets from the `data/raw/` directory.
2. **Data Cleaning:**
   - Handles missing values, removes duplicates, and ensures correct data types for all features.
3. **Exploratory Data Analysis (EDA):**
   - Provides summary statistics, visualizations, and insights into feature distributions and relationships.
4. **Feature Engineering:**
   - Creates new features based on transaction time, frequency, user behavior, and more.
5. **Geolocation Enrichment:**
   - Maps IP addresses to countries to add geographic context to transactions.
6. **Imbalance Handling:**
   - Addresses class imbalance using techniques like SMOTE and undersampling to improve model performance.
7. **Normalization:**
   - Scales and encodes features to prepare data for machine learning models.
8. **Modeling:**
   - Trains and evaluates Logistic Regression and LightGBM models for fraud detection.

## Data

- **Raw Data:**
  - `data/raw/Fraud_Data.csv` (E-commerce fraud)
  - `data/raw/creditcard.csv` (Credit card fraud)
  - `data/raw/IpAddress_to_Country.csv` (IP geolocation)
- **Processed Data:**
  - `data/processed/fraud_one_hot_encoded.csv`
  - `data/processed/credit_minmax_scaled.csv`

## Modeling Approach

- **Logistic Regression:**
  - A simple, interpretable baseline model that provides insight into feature importance and the linear relationships in the data.
- **LightGBM:**
  - A powerful gradient boosting framework that handles large datasets efficiently and captures complex, non-linear patterns.
- **Evaluation Metrics:**
  - Models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and AUC-PR (Precision-Recall Curve), with a focus on minimizing false negatives (missed frauds).

## Sample Results

*Below is an example of model evaluation output (replace with your actual results):*

```
Logistic Regression:
Accuracy: 0.95
Precision: 0.80
Recall: 0.72
F1-score: 0.76
ROC-AUC: 0.91
AUC-PR: 0.85

LightGBM:
Accuracy: 0.97
Precision: 0.88
Recall: 0.81
F1-score: 0.84
ROC-AUC: 0.95
AUC-PR: 0.90
```

Generated plots and figures are saved in the `figures/` directory for further analysis and reporting, e.g.:
- `Univariante_Ip_Address.png`
- `Correlation_Bivariant_heatmap.png`
- `credit_card_class_distribution.png`
- `transaction_count_by_country.png`
- `fraud_data_class_distribution.png`

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── creditcard.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/
│       ├── fraud_one_hot_encoded.csv
│       └── credit_minmax_scaled.csv
├── figures/                    # Generated plots and figures
│   └── ... (see above)
├── model/                      # Trained models
│   ├── credit_lgbm_model.joblib
│   ├── fraud_lgbm_model.joblib
│   ├── credit_logreg_model.joblib
│   └── fraud_logreg_model.joblib
├── notebooks/
│   ├── Preprocessing_exploration.ipynb
│   └── model_exploration.ipynb
├── scripts/
│   ├── main.py                 # Main pipeline script
│   ├── logistic_regression.py  # Logistic Regression model
│   └── lightgbm_model.py       # LightGBM model
├── src/
│   ├── data_loading.py         # DataLoader: Loads datasets
│   ├── data_cleaning.py        # DataCleaner: Handles missing values, duplicates, etc.
│   ├── feature_engineering.py  # FeatureEngineer: Creates time-based and frequency features
│   ├── geolocation.py          # GeolocationProcessor: Enriches data with country info
│   ├── imbalance_handling.py   # ImbalanceHandler: Handles class imbalance (SMOTE, etc.)
│   ├── normalization.py        # DataNormalizer: Scales and encodes features
│   ├── eda.py                  # EDA: Analysis and visualizations
│   ├── data_split.py           # Functions for feature/target split and stratified splitting
│   └── __init__.py
├── tests/
│   └── test_pipeline.py        # Unit tests for core pipeline components
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Requirements

- Python 3.8+
- All dependencies are pinned in `requirements.txt` for reproducibility.

```
pandas==1.2.0
numpy==1.19.0
matplotlib==3.3.3
seaborn==0.11.1
scikit-learn==0.24.0
imbalanced-learn==0.8.0
ipython==7.19.0
jupyter==1.0.0
pytest==6.2.0
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/fraud-detection-for-e-commerce-and-bank-transactions.git
   cd fraud-detection-for-e-commerce-and-bank-transactions
   ```

2. **Create a virtual environment (recommended for reproducibility)**
   ```bash
   python -m venv venv
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Main Pipeline

The main pipeline in `scripts/main.py` orchestrates the entire data preprocessing workflow.
```bash
python scripts/main.py
### Run Jupyter Notebooks for Interactive Analysis

You can interactively explore the data, preprocessing steps, and model results using the provided Jupyter notebooks.

**To launch the notebooks:**
# or, to open a specific notebook directly:
jupyter notebook notebooks/Preprocessing_exploration.ipynb
jupyter notebook notebooks/model_exploration





```

### Train and Evaluate Models

You can run the modeling scripts directly to train and evaluate the models on the preprocessed data.

**Logistic Regression:**
```bash
python scripts/logistic_regression.py
```

**LightGBM:**
```bash
python scripts/lightgbm_model.py
```

### Explore the Data Interactively

Open the Jupyter notebooks for detailed exploration:
```bash
jupyter notebook notebooks/Preprocessing_exploration.ipynb
jupyter notebook notebooks/model_exploration.ipynb
```

### Run Unit Tests

```bash
pytest tests/
```

## How to Use the Data Split Module

To separate features and target, and perform a stratified train-test split, use:
```python
from src.data_split import separate_features_and_target, stratified_train_test_split
X, y = separate_features_and_target(df, target_col='class')  # or 'Class' for creditcard
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)
```

## Troubleshooting

- **ImportError: cannot import name 'DataSplitter'**
  - Use the new function-based API from `src/data_split.py` (see above).
- **FileNotFoundError: ...csv**
  - Ensure the required data file exists in the correct directory. See the Data section above for expected files.
- **ImportError: cannot import name 'LightGBMClassifier'**
  - Use `LGBMClassifier` from the `lightgbm` package, not from your script.
- For Jupyter notebook issues, ensure all dependencies are installed and the kernel is set to your virtual environment.

## Contributing

We welcome contributions from the community! To contribute:
- Fork the repository and create your branch from `main`.
- Ensure your code follows the existing style and includes appropriate tests.
- Submit a pull request with a clear description of your changes.
- For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.



