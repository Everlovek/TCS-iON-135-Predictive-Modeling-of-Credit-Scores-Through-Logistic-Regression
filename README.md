# TCS-iON-135-Predictive-Modeling-of-Credit-Scores-Through-Logistic-Regression
Predictive credit scoring system using Logistic Regression on Apache Spark (PySpark MLlib), trained on 150,000 borrower records to classify default risk with **93.23% accuracy** and **AUC-ROC of 0.7943**.

##  Project Overview

This project implements an end-to-end machine learning pipeline for **credit default prediction**, built as part of an industry internship with **Tata Consultancy Services (TCS)**. The goal is to predict the probability that a borrower will experience serious financial distress within two years, enabling lenders to make faster, more consistent, and explainable credit decisions.

The model is designed with three core business requirements in mind:
 **Interpretability** - results must be explainable to non-technical stakeholders
 **Speed** - inference must work in near real-time for loan applications
 **Cost-effectiveness** - minimal compute resources via distributed computing

##  Dataset
 **Source:** [Give Me Some Credit - Kaggle (2011)](https://www.kaggle.com/c/GiveMeSomeCredit)
 **Size:** 150,000 borrower records, 10 financial features
 **Target variable:** `SeriousDlqin2yrs` - whether a borrower defaulted within 2 years (binary: 0/1)
 **Class imbalance:** ~93% non-default, ~7% default

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Distributed Computing | Apache Spark 4.0.2, PySpark MLlib |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |
| Environment | Google Colab |
| IDE | Jupyter Notebook |

##  Project Structure

```
├── data/
│   └── cs-training.csv           # Raw dataset (from Kaggle)
├── notebooks/
│   └── predictive_modeling.ipynb   # Main Colab notebook
├── model/
│   └── logit_model/              # Saved PySpark LogisticRegression model
├── outputs/
│   ├── plots/                    # EDA and evaluation charts
│   └── inference_results          # Predictions on new borrower profiles
└── README.md
```

### Methodology

The project follows a four-stage pipeline:

### 1. Data Preparation
Loaded dataset into PySpark DataFrame via SparkSession
Handled missing values (imputation for `MonthlyIncome`, `NumberOfDependents`)
Outlier capping using IQR-based thresholds
Feature assembly using `VectorAssembler` and scaling with `StandardScaler`
Train/validation split: **80% / 20%**

### 2. Model Building
Built a PySpark ML Pipeline: `VectorAssembler - StandardScaler - LogisticRegression`
Analyzed feature coefficients and intercept for interpretability
Identified key predictors by coefficient magnitude

### 3. Evaluation & Tuning
Primary metric: **AUC-ROC** (via `BinaryClassificationEvaluator`)
Secondary metrics: Accuracy, Precision, Recall, F1-Score
Decision threshold tuned to **0.3** to improve sensitivity for default detection
Fine-tuned hyperparameters: `regParam`, `elasticNetParam`, `maxIter`

### 4. Deployment & Inference
Model saved and reloaded using `LogisticRegressionModel.save/load`
Inference run on 3 synthetic borrower profiles with credit decision output (approve/decline, low/medium/high risk)

### Results
| Metric | Value |
|---|---|
| AUC-ROC | **0.7943** |
| Accuracy | **93.23%** |
| Decision Threshold | 0.3 |

**Top Predictors (by coefficient weight):**
1. Revolving Utilization of Unsecured Lines
2. Number of Times 90 Days Late
3. Age of Borrower
4. Frequency of Payments (30–59 days past due)

## Key Features
Distributed ML pipeline built entirely on **PySpark MLlib** - scalable to millions of records
Custom **decision threshold (0.3)** to prioritize recall for high-risk borrowers
Full **exploratory data analysis** with correlation heatmaps, class distribution charts, and outlier plots
**Feature importance** derived from logistic regression coefficients
**Inference on new borrowers** with interpretable credit category output
##Getting Started

### Prerequisites
```bash
Python 3.10+
Apache Spark 4.0.2
PySpark MLlib
```

### Run on Google Colab
1. Open the notebook directly: [▶ Launch Notebook on Google Colab](https://colab.research.google.com/drive/1M2Qt5LePN4vu_HPGp_OoprD_iTepQXnO?usp=drive_link)
2. Upload `cs-training.csv` to the Colab session storage or mount Google Drive
3. Run all cells sequentially

### Run Locally
```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebooks/predictive_modeling.ipynb
```

## Academic Context

| Field | Details |
|---|---|
| Institution | Vishwakarma University |
| Course | Internship: TCS iON 135 |
| Programme | FYMSC Statistics-Data Science |
| Company | Tata Consultancy Services (TCS) |
| Duration | 20 Apr 2026 - 03 May 2026 |
| Total Effort | 135 hours |

## References
- Kaggle. (2011). *Give Me Some Credit* competition dataset.
- Apache Spark Documentation : MLlib: [spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html)
- Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley.
- Hand, D.J. & Henley, W.E. (1997). Statistical classification methods in consumer credit scoring. *Journal of the Royal Statistical Society.*

## License

This project was developed for academic and internship purposes under TCS iON. The dataset is publicly available via Kaggle

*Submitted by: **Everlove Fortitude Kavayi** (PRN: 31250500043)*
