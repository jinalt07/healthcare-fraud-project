**Healthcare Provider Fraud Detection**

**Project Overview**

This project builds an end-to-end healthcare provider fraud detection workflow using provider-level, claims-level, and beneficiary-level data.

The goal is to identify providers with elevated potential fraud risk based on billing behavior, claim patterns, patient demographics, chronic-condition burden, reimbursement behavior, and rule-based risk indicators.

The model is intended to be used as a fraud-risk screening tool, not as proof of fraud.

**Project Objective**

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

**Dataset Description**

The project uses three main types of data:

1. **Provider data**  
   Contains provider-level billing and utilization summaries.

2. **Claims data**  
   Contains inpatient and outpatient claim-level information.

3. **Beneficiary data**  
   Contains patient demographic and chronic-condition information.

The actual test dataset does not contain the fraud label, so the trained model is used to generate fraud-risk predictions.

**Project Workflow**

**1. Data Loading**

Raw training and test datasets were loaded from the project data folders.

**2. Data Cleaning**

The cleaning process included:

- Checking missing values
- Checking duplicate records
- Converting claim dates to datetime format
- Handling missing deductible values
- Encoding chronic-condition variables
- Encoding renal disease indicators
- Validating claim durations
- Checking negative reimbursement and deductible values

**3. Feature Engineering**

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
