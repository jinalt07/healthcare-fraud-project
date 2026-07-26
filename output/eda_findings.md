# Exploratory Data Analysis Findings

## Dataset Overview

- The dataset contains 5,410 healthcare providers.
- 506 providers, or approximately 9.35%, are labeled as potential fraud.
- The dataset is imbalanced, with substantially more non-fraud providers than fraud-labeled providers.

## Billing and Reimbursement Patterns

- Fraud-labeled providers submitted substantially more claims on average than non-fraud providers:
  - Non-fraud providers: approximately 70 total claims per provider.
  - Fraud-labeled providers: approximately 421 total claims per provider.
- Fraud-labeled providers also received higher total reimbursements on average.
- Claims per patient and reimbursement per patient should be considered alongside total values because larger providers naturally submit more claims and treat more patients.

## Claim Duration and High-Cost Claims

- Fraud-labeled providers had longer claim durations:
  - Average maximum claim duration: 26.97 days for fraud-labeled providers versus 15.07 days for non-fraud providers.
- Fraud-labeled providers had more high-cost claims in absolute terms:
  - Fraud-labeled providers: 22.06 high-cost claims on average.
  - Non-fraud providers: 3.68 high-cost claims on average.
- The percentage of high-cost claims was only moderately higher for fraud-labeled providers:
  - Fraud-labeled providers: 5.87%.
  - Non-fraud providers: 5.25%.

## Provider Size Effects

- Total reimbursement, deductible amount, inpatient claims, patient counts, and chronic-disease patient counts had strong positive correlations with the fraud label.
- These relationships largely appear to reflect provider size: larger providers naturally have more claims, patients, reimbursements, and chronic-condition counts.
- Therefore, size-adjusted variables such as claims per patient, reimbursement per patient, claim ratios, and disease rates are important for modeling.

## Redundant Features

- Several feature groups are duplicates or near-duplicates:
  - `total_claims` and `unique_claims`
  - `unique_patients`, `total_gender_patients`, and `total_race_patients`
  - `inpatient_claim_ratio` and `outpatient_claim_ratio`
  - `male_patient_percentage` and `female_patient_percentage`
- Raw disease counts are strongly correlated with provider size. Disease rates may be more useful than keeping every raw disease-count feature.

## Limitations

- A potential-fraud label is not proof that a provider committed fraud.
- The data does not include diagnosis codes, procedure codes, or service-level billing codes.
- Therefore, the project cannot directly test upcoding, unnecessary procedures, or diagnosis-procedure mismatches.
- The model should be positioned as a provider risk-screening tool that helps prioritize cases for review.