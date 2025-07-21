# Fraud Detection for E-Commerce and Bank Transactions

<!-- If you have a logo, place it here -->
<!-- ![Project Logo](figures/logo.png) -->

## Overview

Fraudulent transactions are a major challenge for both e-commerce platforms and financial institutions, leading to significant financial losses and eroding customer trust. This project aims to build a robust, end-to-end pipeline for detecting fraudulent activities in transactional data. By leveraging advanced data preprocessing, feature engineering, and state-of-the-art machine learning models, this solution helps organizations identify and prevent fraud in real time, reducing risk and improving operational efficiency.

**Key Features:**
- Modular Python codebase for easy extension and maintenance
- Comprehensive EDA with both univariate and bivariate analysis
- Automated handling of missing data, outliers, and class imbalance
- Geolocation enrichment using IP-to-country mapping
- Feature engineering for time, frequency, and user behavior
- Implementation of Logistic Regression and LightGBM models
- Jupyter notebooks for interactive exploration and reproducibility
- Unit tests for core pipeline components

## Pipeline Description

The pipeline is designed to be modular and extensible, covering all critical stages of a modern fraud detection workflow:

1. **Data Loading:**
   - Loads e-commerce, credit card, and IP-to-country datasets from the `data/` directory.
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

## Modeling Approach

- **Logistic Regression:**
  - A simple, interpretable baseline model that provides insight into feature importance and the linear relationships in the data.
- **LightGBM:**
  - A powerful gradient boosting framework that handles large datasets efficiently and captures complex, non-linear patterns.
- **Evaluation Metrics:**
  - Models are evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, with a focus on minimizing false negatives (missed frauds).

## Sample Results

*Below is an example of model evaluation output (replace with your actual results):*

```
Logistic Regression:
Accuracy: 0.95
Precision: 0.80
Recall: 0.72
F1-score: 0.76
ROC-AUC: 0.91

LightGBM:
Accuracy: 0.97
Precision: 0.88
Recall: 0.81
F1-score: 0.84
ROC-AUC: 0.95
```

Generated plots and figures are saved in the `figures/` directory for further analysis and reporting.

## Project Structure

```
.
├── data/                       # Data directory (raw and processed data)
├── figures/                    # Directory for generated plots and figures
├── model/                      # Directory for trained models
│   └── logreg_credit_20250721_163443.joblib
├── notebooks/
│   ├── Preprocessing_exploration.ipynb # Jupyter notebook for data preprocessing exploration
│   └── model_exploration.ipynb       # Jupyter notebook for model exploration
├── scripts/
│   ├── main.py                 # Main pipeline script orchestrating all steps
│   ├── logistic_regression.py  # Script for Logistic Regression model
│   └── lightgbm_model.py       # Script for LightGBM model
├── src/                        # Source code modules
│   ├── data_loading.py         # DataLoader: Loads datasets
│   ├── data_cleaning.py        # DataCleaner: Handles missing values, duplicates, etc.
│   ├── feature_engineering.py  # FeatureEngineer: Creates time-based and frequency features
│   ├── geolocation.py          # GeolocationProcessor: Enriches data with country info
│   ├── imbalance_handling.py   # ImbalanceHandler: Handles class imbalance (SMOTE, etc.)
│   ├── normalization.py        # DataNormalizer: Scales and encodes features
│   ├── eda.py                  # EDA: Analysis and visualizations
│   └── __init__.py
├── tests/
│   └── test_pipeline.py        # Unit tests for core pipeline components
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all Python dependencies.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/fraud-detection-for-e-commerce-and-bank-transactions.git
   cd fraud-detection-for-e-commerce-and-bank-transactions
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Data

- Place your raw data files in the `data/` directory. The scripts are configured to look for datasets there.

## Troubleshooting

- If you encounter issues with dependencies, ensure your Python version matches the requirements and that your virtual environment is activated.
- For Jupyter notebook issues, ensure all dependencies are installed and the kernel is set to your virtual environment.

## Contributing

We welcome contributions from the community! To contribute:
- Fork the repository and create your branch from `main`.
- Ensure your code follows the existing style and includes appropriate tests.
- Submit a pull request with a clear description of your changes.
- For major changes, please open an issue first to discuss what you would like to change.


## License

This project is licensed under the MIT License.



