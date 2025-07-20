# Fraud Detection for E-Commerce and Bank Transactions

<!-- If you have a logo, place it here -->
<!-- ![Project Logo](figures/logo.png) -->

## Overview

This project provides a robust, modular pipeline for detecting fraudulent transactions in e-commerce and banking datasets. The workflow covers all key stages of a modern data science project, including data loading, cleaning, exploratory data analysis (EDA), feature engineering, geolocation enrichment, class imbalance handling, normalization, and lays the groundwork for advanced modeling and explainability.

**Key Features:**
- Modular Python codebase for easy extension and maintenance
- Comprehensive EDA with both univariate and bivariate analysis
- Automated handling of missing data, outliers, and class imbalance
- Geolocation enrichment using IP-to-country mapping
- Feature engineering for time, frequency, and user behavior
- Ready for integration with machine learning models
- Jupyter notebook for interactive exploration and reproducibility
- Unit tests for core pipeline components

## Project Structure

```
.
├── src/                # Source code modules
│   ├── data_loading.py         # DataLoader: Loads e-commerce, IP-to-country, and credit card datasets
│   ├── data_cleaning.py        # DataCleaner: Handles missing values, removes duplicates, converts data types
│   ├── feature_engineering.py  # FeatureEngineer: Creates time-based and frequency features
│   ├── geolocation.py          # GeolocationProcessor: Enriches data with country info from IP addresses
│   ├── imbalance_handling.py   # ImbalanceHandler: Handles class imbalance (SMOTE, undersampling)
│   ├── normalization.py        # DataNormalizer: Scales and encodes features
│   ├── eda.py                  # EDA: Summary stats, univariate/bivariate analysis, visualizations
│   └── __init__.py
├── scripts/
│   └── main.py                 # Main pipeline script orchestrating all steps
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebook for EDA and interactive analysis
├── tests/
│   └── test_pipeline.py        # Unit tests for core pipeline components
├── data/                       # Data directory (raw and processed data)
├── figures/                    # Directory for generated plots and figures
├── Report.md                   # Project report (methodology, results, recommendations)
├── README.md                   # This file
└── .gitignore
```

## Installation & Requirements

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd fraud-detection-for-e-commerce-and-bank-transactions
   ```

2. **Install dependencies**
   - Create a virtual environment (recommended)
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   *(If `requirements.txt` is missing, install: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, ipython, jupyter, pytest)*

## Usage

### Run the Main Pipeline

```bash
python scripts/main.py
```

### Explore the Data Interactively

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Run Unit Tests

```bash
pytest tests/test_pipeline.py
```

## Example

```python
from src.eda import EDA
eda = EDA()
eda.univariate_analysis(df)  # Plots up to 2 univariate distributions
eda.bivariate_analysis(df, target_col='class')  # Plots up to 2 bivariate plots
```

## Results and Reporting

- See `Report.md` for methodology, results, and recommendations.
- Generated plots and figures are saved in the `figures/` directory.

## Data

- Place your raw data files in the `data/` directory.
- Data sources: [Specify your data sources here, e.g., Kaggle, UCI, etc.]

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements or bug fixes.

## License

[Specify your license here, e.g., MIT License.]

