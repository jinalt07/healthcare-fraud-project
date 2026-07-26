from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"
OUTPUT_FOLDER = PROJECT_ROOT / "output" / "models"

final_training_data_file = PROCESSED_FOLDER / "final_training_dataset.csv"

final_training_data = pd.read_csv(final_training_data_file, low_memory= False)

print("\nFinal Data Set")
print(final_training_data.head())

print(final_training_data.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float.format", "{:,.2f}".format)

##========================================================================================================================
## 90th percentile threshlods
##=========================================================================================================================


risk_threshold = (final_training_data[["total_claims", "total_reimbursement", "claim_per_patient", "reimbursement_per_patient", "inpatient_claim_ratio", "high_cost_claim_percentage", "max_claim_duration_days"]].quantile(0.90))

print("\n90th Percentile Risk Threshold")
print(risk_threshold.round(2).to_string())


##=========================================================================================================================
## Risk Flags
##=========================================================================================================================

final_training_data["high_total_claims_flag"] = (final_training_data["total_claims"] >= risk_threshold["total_claims"]).astype(int)

final_training_data["high_total_reimbursement_flag"] = (final_training_data["total_reimbursement"] >= risk_threshold["total_reimbursement"]).astype(int)

final_training_data["high_claim_per_patient_flag"] = (final_training_data["claim_per_patient"] >= risk_threshold["claim_per_patient"]).astype(int)

final_training_data["high_reimbursement_per_patient_flag"] = (final_training_data["reimbursement_per_patient"] >= risk_threshold["reimbursement_per_patient"]).astype(int)

final_training_data["high_inpatient_claim_ratio_flag"] = (final_training_data["inpatient_claim_ratio"] >= risk_threshold["inpatient_claim_ratio"]).astype(int)

final_training_data["high_cost_claim_percentage_flag"] = (final_training_data["high_cost_claim_percentage"] >= risk_threshold["high_cost_claim_percentage"]).astype(int)

final_training_data["high_max_claim_duration_flag"] = (final_training_data["max_claim_duration_days"] >= risk_threshold["max_claim_duration_days"]).astype(int)

##=========================================================================================================================
## Total Risk Score
##=========================================================================================================================

final_training_data["risk_score"] = (final_training_data[["high_total_claims_flag", "high_total_reimbursement_flag", "high_claim_per_patient_flag", "high_reimbursement_per_patient_flag", "high_inpatient_claim_ratio_flag", "high_cost_claim_percentage_flag", "high_max_claim_duration_flag"]].sum(axis=1))

print("\nRisk Score Distribution")

print(final_training_data["risk_score"].value_counts().sort_index().to_string())

##========================================================================================================================
## Risk Level
##========================================================================================================================

def risk_level (risk_score):
    if risk_score <=1:
        return "Low"
    elif risk_score <=3:
        return "Medium"
    else:
        return "High"
    
final_training_data["risk_level"] = (final_training_data["risk_score"].apply(risk_level))

print("\nRisk Level")
print(final_training_data["risk_level"].value_counts().to_string())

##=========================================================================================================================
## Risk level vs Fraud label
##=========================================================================================================================

risk_fraud_comparison = pd.crosstab(final_training_data["risk_level"], final_training_data["PotentialFraud"], margins= True)

print("\nRisk Level vs Fraud Label")
print(risk_fraud_comparison.to_string())

risk_fraud_percentage = pd.crosstab(final_training_data["risk_level"], final_training_data["PotentialFraud"], normalize= "index") * 100

print("\nFraud Percentage by Risk Level")
print(risk_fraud_percentage.round(2).to_string())

##=========================================================================================================================
## Save Provider Risk Level
##=========================================================================================================================

provider_risk_table = final_training_data[["Provider", "PotentialFraud", "PotentialFraudFlag", "risk_score", "risk_level", "high_total_claims_flag", "high_total_reimbursement_flag", "high_claim_per_patient_flag", "high_reimbursement_per_patient_flag", "high_inpatient_claim_ratio_flag", "high_cost_claim_percentage_flag", "high_max_claim_duration_flag"]].copy()

provider_risk_table_file = OUTPUT_FOLDER / "provider_risk_table.csv"

provider_risk_table.to_csv(provider_risk_table_file, index= False)

print("\nProvider risk level table")
print(provider_risk_table.head().to_string())


##=========================================================================================================================
## High risk providers
##=========================================================================================================================

high_risk_providers = (final_training_data[final_training_data["risk_level"] == "High"].sort_values(by = ["risk_score", "total_reimbursement", "total_claims"], ascending= False)[["Provider", "PotentialFraud", "risk_score", "risk_level", "total_claims", "total_reimbursement", "claim_per_patient", "reimbursement_per_patient", "inpatient_claim_ratio", "high_cost_claim_percentage", "max_claim_duration_days"]])

high_risk_providers_file = OUTPUT_FOLDER / "high_risk_providers.csv"

high_risk_providers.to_csv(high_risk_providers_file, index = False)

print("\nHigh Risk Provider")
print(high_risk_providers.shape)

##=========================================================================================================================
## Risk threshold 
##=========================================================================================================================

risk_threshold_table = (risk_threshold.reset_index().rename(columns={"index": "risk_feature", 0.90: "threshold_value"}))

risk_threshold_file = OUTPUT_FOLDER / "risk_threshold.csv"

risk_threshold_table.to_csv(risk_threshold_file, index = False)

print("\nRisk Threshold")
print(risk_threshold)