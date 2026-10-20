<h1 style="border-bottom: none;">Healthcare Provider Fraud Detection</h1>

<h2 style="border-bottom: none;">Project Overview</h2>

This project builds an end-to-end healthcare provider fraud detection workflow using provider-level, claims-level, and beneficiary-level data.

The goal is to identify providers with elevated potential fraud risk based on billing behavior, claim patterns, patient demographics, chronic-condition burden, reimbursement behavior, and rule-based risk indicators.

The model is intended to be used as a fraud-risk screening tool, not as proof of fraud.

---

## Project Objective

The objective of this project is to predict whether a healthcare provider is likely to be associated with potential fraud using historical claims and beneficiary data.

The project covers:

- Data loading
- Data cleaning
- Feature engineering
- Exploratory data analysis
- Rule-based risk scoring
- Machine learning model training
- Model evaluation
- Actual test data prediction
- Final visualizations
- PDF project report generation

---

## Dataset Description

The project uses three main types of data:

1. Provider data  
   Contains provider-level billing and utilization summaries.

2. Claims data  
   Contains inpatient and outpatient claim-level information.

3. Beneficiary data  
   Contains patient demographic and chronic-condition information.

The actual test dataset does not contain the fraud label, so the trained model is used to generate fraud-risk predictions.

---

## Project Workflow

### 1. Data Loading

Raw training and test datasets were loaded from the project data folders.

### 2. Data Cleaning

The cleaning process included:

- Checking missing values
- Checking duplicate records
- Converting claim dates to datetime format
- Handling missing deductible values
- Encoding chronic-condition variables
- Encoding renal disease indicators
- Validating claim durations
- Checking negative reimbursement and deductible values

### 3. Feature Engineering

Provider-level features were created from claims and beneficiary data.

Feature groups included:

- Billing features
- Reimbursement features
- Claim frequency features
- Claim duration features
- Patient demographic features
- Chronic-condition burden features
- High-cost claim features
- Inpatient/outpatient claim ratio features

### 4. Exploratory Data Analysis

EDA was performed to compare fraud and non-fraud providers across billing, utilization, reimbursement, demographic, chronic-condition, and fraud-risk features.

### 5. Rule-Based Risk Scoring

A rule-based provider risk score was created using selected high-risk indicators.

Providers were assigned risk levels:

- Low risk
- Medium risk
- High risk

The risk score helped interpret and compare model predictions.

### 6. Machine Learning

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The models were evaluated using:

- Fraud precision
- Fraud recall
- Fraud F1-score
- ROC-AUC
- PR-AUC

---

## Selected Model

The final selected model was:

**Gradient Boosting Classifier**

Gradient Boosting was selected because it provided strong overall fraud-screening performance, especially on fraud recall and fraud F1-score.

Validation performance:

| Metric | Score |
|---|---:|
| Fraud Precision | 0.51 |
| Fraud Recall | 0.90 |
| Fraud F1-score | 0.65 |
| ROC-AUC | 0.9674 |
| PR-AUC | 0.7825 |

The model caught most fraud-labeled providers in the validation dataset while maintaining reasonable precision.

---

## Actual Test Prediction Results

The trained Gradient Boosting model was applied to the actual test provider dataset.

Test prediction summary:

| Metric | Value |
|---|---:|
| Total test providers | 1,353 |
| Predicted potential fraud providers | 227 |
| Predicted not potential fraud providers | 1,126 |
| High risk providers | 16 |
| Medium risk providers | 168 |
| Low risk providers | 1,169 |

Prediction by risk level:

| Risk Level | Not Potential Fraud | Potential Fraud |
|---|---:|---:|
| High | 0 | 16 |
| Medium | 42 | 126 |
| Low | 1,084 | 85 |

---

## Key Visualizations

The project includes the following visualizations:

- Predicted fraud status distribution
- Risk level distribution
- Fraud prediction by risk level
- Fraud probability distribution
- Top 20 providers by fraud probability
- Top 20 model feature importance
- Model performance comparison

All charts are saved in:

```text
output/charts/
```

---

## Final Outputs

Important output files:

```text
output/models/gradient_boosting_model.pkl
output/models/X_test_actual.csv
output/predictions/test_provider_predictions.csv
output/reports/final_healthcare_fraud_detection_report_v2.pdf
```

---

## Project Structure

```text
healthcare-fraud-project/
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── output/
│   ├── charts/
│   ├── models/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── 01_load_data.py
│   ├── 02_clean_data.py
│   ├── 03_merge_claim_beneficiary.py
│   ├── 04_create_provider_features.py
│   ├── 05_create_fraud_risk_features.py
│   ├── 06_build_final_training_dataset.py
│   ├── 07_exploratory_analysis.py
│   ├── 08_create_provider_risk_levels.py
│   ├── 09_prepare_machine_learning_data.py
│   ├── 10_train_models.py
│   ├── 11_evaluate_models.py
│   ├── 12_prepare_test_data.py
│   ├── 13_predict_test_data.py
│   ├── 14_create_final_visualizations.py
│   └── 15_project_report.py
│
└── README.md
```

---

## Limitations

This project identifies potential fraud risk, not confirmed fraud.

The model should not be treated as legal or investigative proof of fraud. It should be used only as a screening tool to prioritize providers for further review.

The dataset does not include investigation outcomes, legal findings, or detailed diagnosis/procedure-code level fraud patterns. Therefore, the model is limited to the available provider, claims, reimbursement, demographic, and chronic-condition features.

---

## Conclusion

This project demonstrates a complete healthcare fraud detection workflow from raw data to final test predictions and reporting.

The final Gradient Boosting model produced provider-level fraud-risk predictions and helped identify providers with elevated potential fraud probability.

The project shows how machine learning and rule-based risk scoring can be combined to support healthcare fraud-risk screening.
