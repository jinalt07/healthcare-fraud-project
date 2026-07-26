##=========================================================================================================================
## Load Data
##==========================================================================================================================

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"

final_training_data_file = (PROCESSED_FOLDER / "final_training_dataset.csv")

final_training_data = pd.read_csv(final_training_data_file, low_memory=False)

print(final_training_data.columns.to_list())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float.format", "{:,.2f}".format)


##===========================================================================================================================
## Fraud Distribution
##===========================================================================================================================

print("\nFraud Counts")
print(final_training_data["PotentialFraud"].value_counts())

print("\nPotential Fraud Percentage")
print((final_training_data["PotentialFraud"].value_counts(normalize=True)*100).round(2))


##=========================================================================================================================
## Billing Features and Fraud Compare
##=========================================================================================================================

fruad_billing_mean = (final_training_data.groupby("PotentialFraud")[["total_claims", "claim_per_patient", "total_reimbursement", "avg_reimbursement", "reimbursement_per_patient", "avg_deductible", "inpatient_claim_ratio", "high_cost_claim_percentage", "median_claim_duration_days", "max_claim_duration_days"]].mean().round(2)) 

print("\nAvg Billing Features by Fraud Label")
print(fruad_billing_mean)


##=========================================================================================================================
## Patient Demographics and Fraud Compare
##=========================================================================================================================

fraud_demographic_mean = (final_training_data.groupby("PotentialFraud")[["unique_patients", "avg_patient_age", "median_patient_age", "male_patient_percentage", "female_patient_percentage", "race_1_percentage", "race_2_percentage", "race_3_percentage", "race_5_percentage"]].mean().round(2))

print("\nAvg Demographic Features by Fraud Label")
print(fraud_demographic_mean)


##==========================================================================================================================        
## Patient Chronic Condition and Fraud Compare
##==========================================================================================================================

fraud_chronic_mean = (final_training_data.groupby("PotentialFraud")[["avg_chronic_condition_count", "median_chronic_condition_count", "alzheimer_patient_rate", "heartfailure_patient_rate", "kidney_disease_patient_rate", "cancer_patient_rate", "obstructive_pulmonary_patient_rate", 
                                                                     "depression_patient_rate", "diabetes_patient_rate", "ischemic_heart_patient_rate", "osteoporosis_patient_rate", "rheumatoid_arthritis_patient_rate", "stroke_patient_rate", "renal_disease_patient_rate"]].mean().round(2))

print("\nAvg Chronic Features by fraud Label")
print(fraud_chronic_mean)


##=========================================================================================================================
## Patient Fraud-Risk Features and Fraud Compare
##=========================================================================================================================

fraud_risk_mean = (final_training_data.groupby("PotentialFraud")[["median_claim_duration_days", "max_claim_duration_days", "high_cost_claim_count", "high_cost_claim_percentage"]].mean().round(2)) 

print("\nAvg Fraud Risk Features by Fraud Label")
print(fraud_risk_mean)


##=========================================================================================================================
## Fruad- Lable distribution charts
##=========================================================================================================================/

## Chart 1 - Potential Fraud Counts

import matplotlib.pyplot as plt

OUTPUT_FOLDER = PROJECT_ROOT / "output"

CHARTS_FOLDER = OUTPUT_FOLDER / "charts"

CHARTS_FOLDER.mkdir(parents= True, exist_ok= True)

fraud_counts = (final_training_data["PotentialFraud"].value_counts())

plt.figure(figsize=(8,5))

plt.bar(fraud_counts.index, fraud_counts.values)

plt.title("Provider Fraud Label Distribution")

plt.xlabel("Potential Fraud Label")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "fraud_distribution.png", dpi=300)

plt.show()

## Chart 2 - Avg Total Claims

average_total_claims = (final_training_data.groupby("PotentialFraud")["total_claims"].mean())

plt.figure(figsize=(8,5))

plt.bar(average_total_claims.index, average_total_claims.values)

plt.title("Average Total Claims by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average Total Claims per Provider")

plt.tight_layout()

plt.savefig( CHARTS_FOLDER / "average_total_claims_by_fraud_label.png", dpi= 300)

plt.show()

## Chart 3 - Avg Total Reimbursement 

average_total_reimbursement = (final_training_data.groupby("PotentialFraud")["total_reimbursement"].mean())

plt.figure(figsize=(8, 5))

plt.bar(average_total_reimbursement.index,average_total_reimbursement.values)

plt.title("Average Total Reimbursement by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average Total Reimbursement per Provider")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "average_total_reimbursement_by_fraud_label.png",dpi=300)

plt.show()

## Chart 4 - Average Reimbursement per Patient

average_reimbursement_per_patient = (final_training_data.groupby("PotentialFraud")["reimbursement_per_patient"].mean())

plt.figure(figsize=(8, 5))

plt.bar(average_reimbursement_per_patient.index,average_reimbursement_per_patient.values)

plt.title("Average Reimbursement per Patient by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average Reimbursement per Patient")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "average_reimbursement_per_patient_by_fraud_label.png",dpi=300)

plt.show()

## Chart 5 - Avg Inpatient Claim Ratio 

average_inpatient_claim_ratio = (final_training_data.groupby("PotentialFraud")["inpatient_claim_ratio"].mean())

plt.figure(figsize=(8, 5))

plt.bar(average_inpatient_claim_ratio.index,average_inpatient_claim_ratio.values)

plt.title("Average Inpatient Claim Ratio by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average Inpatient Claim Ratio")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "average_inpatient_claim_ratio_by_fraud_label.png",dpi=300)

plt.show()

## Chart 6 - Avg High-Cost Claim Percentage

average_high_cost_claim_percentage = (final_training_data.groupby("PotentialFraud")["high_cost_claim_percentage"].mean())

plt.figure(figsize=(8, 5))

plt.bar(average_high_cost_claim_percentage.index, average_high_cost_claim_percentage.values)

plt.title("Average High-Cost Claim Percentage by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average High-Cost Claims (%)")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "average_high_cost_claim_percentage_by_fraud_label.png",dpi=300)

plt.show()

## Chart 7 - Avg Max Claim Duration

average_max_claim_duration = (final_training_data.groupby("PotentialFraud")["max_claim_duration_days"].mean())

plt.figure(figsize=(8, 5))

plt.bar(average_max_claim_duration.index,average_max_claim_duration.values)

plt.title("Average Maximum Claim Duration by Fraud Label")

plt.xlabel("Potential Fraud Label")

plt.ylabel("Average Maximum Claim Duration (Days)")

plt.tight_layout()

plt.savefig(CHARTS_FOLDER / "average_max_claim_duration_by_fraud_label.png",dpi=300)

plt.show()

##==========================================================================================================================
## Feature Correlation
##=========================================================================================================================

numeric_data = final_training_data.select_dtypes(include="number")

fraud_correlations = (numeric_data.corrwith(final_training_data["PotentialFraudFlag"]).drop("PotentialFraudFlag").sort_values(key=abs, ascending=False))

print("\nFeatues Correlation with Potential Fraud")
print(fraud_correlations.head(20).round(3).to_string())


##=========================================================================================================================
## Overlapping Features
##=========================================================================================================================

import numpy as np

numeric_features = (final_training_data.select_dtypes(include="number").drop(columns=["PotentialFraudFlag"]))

feature_correlation_matrix = (numeric_features.corr().abs())

upper_triangle = feature_correlation_matrix.where(np.triu(np.ones(feature_correlation_matrix.shape), k=1).astype(bool))

highly_correlated_pairs = (upper_triangle.stack().sort_values(ascending=False))

highly_correlated_pairs = highly_correlated_pairs[highly_correlated_pairs >= 0.90]

print(highly_correlated_pairs.to_string())

##========================================================================================================================
## Target Leakage and Identifier
##======================================================================================================================== 

fraud_related_columns = []

for column_name in final_training_data.columns:
    if "fraud" in column_name.lower():
        fraud_related_columns.append(column_name)

print(fraud_related_columns)