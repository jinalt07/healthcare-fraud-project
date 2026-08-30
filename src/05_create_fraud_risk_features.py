##=========================================================================================================================
## Load Data
##=========================================================================================================================

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed" / "Train"

claims_beneficiary_file = PROCESSED_FOLDER / "claim_beneficiary_clean.csv"

claims_beneficiary = pd.read_csv(claims_beneficiary_file, low_memory= False, parse_dates= ["ClaimStartDt", "ClaimEndDt", "DOB", "DOD"])

print(claims_beneficiary.columns.tolist())


##==========================================================================================================================
## Feature - Claim Duration
##==========================================================================================================================

claims_beneficiary["claims_duration_days"] = (claims_beneficiary["ClaimEndDt"] - claims_beneficiary["ClaimStartDt"]).dt.days

print("\nDescribe Claim duration")
print(claims_beneficiary["claims_duration_days"].describe())

print("\nMissing value")
print(claims_beneficiary["claims_duration_days"].isna().sum())


## Provider level Claim Duration
##=========================================================================================================================

provider_duration_features = (claims_beneficiary.groupby("Provider", as_index= False).agg(median_claim_duration_days = ("claims_duration_days", "median"), max_claim_duration_days = ("claims_duration_days", "max")))
provider_duration_features["median_claim_duration_days"] = (provider_duration_features["median_claim_duration_days"].round(2))

print("\nProvider duration Shape")
print(provider_duration_features.shape)

print("\nProvider duration Head")
print(provider_duration_features.head())


##=========================================================================================================================
##  Feature - High Cost Claim Flag 
##=========================================================================================================================

## Threshhold creation
##=========================================================================================================================

high_cost_thresholds = claims_beneficiary.groupby("claim_type")["InscClaimAmtReimbursed"].quantile(0.95)

print("\nThreshold")
print(high_cost_thresholds)


## High Cost Claim Flag
##==========================================================================================================================

claims_beneficiary["high_cost_threshold"] = (claims_beneficiary["claim_type"].map(high_cost_thresholds))

claims_beneficiary["high_cost_claim_flag"] = (claims_beneficiary["InscClaimAmtReimbursed"] >= claims_beneficiary["high_cost_threshold"]).astype(int)

print("\nCount")
print(claims_beneficiary["high_cost_claim_flag"].value_counts())

print(claims_beneficiary["high_cost_threshold"])

print("\nHigh Cost Claim %")
print(claims_beneficiary.groupby("claim_type")["high_cost_claim_flag"].mean().mul(100).round(2))

## High Cost Claim Feature
##=========================================================================================================================

provider_high_cost_features = (claims_beneficiary.groupby("Provider", as_index= False).agg(high_cost_claim_count = ("high_cost_claim_flag", "sum"), high_cost_claim_percentage= ("high_cost_claim_flag", "mean")))

provider_high_cost_features["high_cost_claim_percentage"] = (provider_high_cost_features["high_cost_claim_percentage"]*100).round(2)

print("\nProvider High Cost Shape")
print(provider_high_cost_features.shape)

print("\nProvider High Cost Head")
print(provider_high_cost_features.head())


##=========================================================================================================================
## Merge Duration and High Cost Features
##=========================================================================================================================

provider_fraud_risk_features = provider_duration_features.merge(provider_high_cost_features, on= "Provider", how= "left", validate= "one_to_one")

print("\nProvider Fraud Rist Features Shape")
print(provider_fraud_risk_features.shape)

print("\nProvider Fraud Risk Features Head")
print(provider_fraud_risk_features.head())


## Save Data
##==========================================================================================================================

provider_fraud_risk_features_file = (PROCESSED_FOLDER / "provider_fraud_risk_features.csv")

provider_fraud_risk_features.to_csv(provider_fraud_risk_features_file, index= False)

print(provider_fraud_risk_features_file.exists())