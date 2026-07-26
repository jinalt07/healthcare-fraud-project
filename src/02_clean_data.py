##=================================================================================================
## Data Cleaning
##=================================================================================================

## Load Data
##================================================================================================
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FOLDER = PROJECT_ROOT / "Data" / "Raw" / "Train"

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"

PROCESSED_FOLDER.mkdir(parents= True, exist_ok =True)

provider_file = TRAIN_FOLDER / "Final_Provider_Train.csv"
claims_file = TRAIN_FOLDER / "All_Claims_Train.csv"
beneficiary_file = TRAIN_FOLDER / "Train_Beneficiarydata-1542865627584.csv"

provider_data = pd.read_csv(provider_file, low_memory= False)
claims_data = pd.read_csv(claims_file, low_memory =False)
beneficiary_data = pd.read_csv(beneficiary_file, low_memory=False)

provider_clean = provider_data.copy()
claims_clean = claims_data.copy()
beneficiary_clean = beneficiary_data.copy()


## Clean Claims_data
##==================================================================================================

## Changing Datatypes str-date Format

print(claims_clean[["ClaimStartDt", "ClaimEndDt"]].head())
claims_clean["ClaimStartDt"] = pd.to_datetime(claims_clean["ClaimStartDt"])
claims_clean["ClaimEndDt"] = pd.to_datetime(claims_clean["ClaimEndDt"])

print("\nChanged date format")
print(claims_clean[["ClaimStartDt", "ClaimEndDt"]].dtypes)

print("\nClaim Start Date Missing Values")
print(claims_clean["ClaimStartDt"].isna().sum())

print("\nClaim End Date Missing Values")
print(claims_clean["ClaimEndDt"].isna().sum())

## Missing values - Replace decutible with Median
print("\nCheck missig values for claim type")
print(claims_clean[claims_clean["DeductibleAmtPaid"].isna()]["claim_type"].value_counts())
claims_clean["DeductibleAmtPaid_missing"] = (claims_clean["DeductibleAmtPaid"].isna().astype(int))
print(claims_clean["DeductibleAmtPaid_missing"].value_counts())

inpatient_deductible_median = claims_clean.loc[claims_clean["claim_type"] == "Inpatient", "DeductibleAmtPaid"].median()

print("\nmedian")
print(inpatient_deductible_median)
claims_clean["DeductibleAmtPaid"] = claims_clean["DeductibleAmtPaid"].fillna(inpatient_deductible_median)

print("\nCheck missing values - after median")
print(claims_clean["DeductibleAmtPaid"].isna().sum())

## Date Check

print("\nCheck if Claim End is before Claim Start")
print((claims_clean["ClaimEndDt"] < claims_clean["ClaimStartDt"]).sum())

## Claim Reimbursement
print("\nNegative Claim Reimbursement")
print((claims_clean["InscClaimAmtReimbursed"] < 0).sum())

## Claim Deductible
print("\n Negative Deductible")
print((claims_clean["DeductibleAmtPaid"] < 0).sum())


## Clean Beneficiary_data
##====================================================================================================

## Changing Datatypes str-date Format

print(beneficiary_clean[["DOB", "DOD"]].head())
beneficiary_clean["DOB"] = pd.to_datetime(beneficiary_clean["DOB"])
beneficiary_clean["DOD"] = pd.to_datetime(beneficiary_clean["DOD"])

print("\nChanges data type")
print(beneficiary_clean[["DOB", "DOD"]].dtypes)

## DOD and DOB check
print("\nDOD before DOB")
print((beneficiary_clean["DOD"].notna() & (beneficiary_clean["DOD"] < beneficiary_clean["DOB"])).value_counts())

## DOB Min Max
print("\nDOB Minimum and Maximum")
print(beneficiary_clean["DOB"].agg(["min", "max"]))

## Encoding Renal Diesease Indicator

print("\nRenal Disease Indicator Values After Cleaning")
print(
    beneficiary_clean["RenalDiseaseIndicator"]
    .value_counts(dropna=False)
)

beneficiary_clean["RenalDiseaseIndicator"] = (beneficiary_clean["RenalDiseaseIndicator"].replace({"Y" : 1, "0": 0}).astype(int))

print(beneficiary_clean["RenalDiseaseIndicator"].value_counts())

## Encoding Chronic Disease
print(beneficiary_clean["ChronicCond_KidneyDisease"].value_counts(dropna=False))

beneficiary_clean["ChronicCond_KidneyDisease"] = (beneficiary_clean["ChronicCond_KidneyDisease"].replace({2:0}))
beneficiary_clean["ChronicCond_Alzheimer"] = (beneficiary_clean["ChronicCond_Alzheimer"].replace({2:0}))
beneficiary_clean["ChronicCond_Heartfailure"] = (beneficiary_clean["ChronicCond_Heartfailure"].replace({2:0}))
beneficiary_clean["ChronicCond_Cancer"] = (beneficiary_clean["ChronicCond_Cancer"].replace({2: 0}))
beneficiary_clean["ChronicCond_ObstrPulmonary"] = (beneficiary_clean["ChronicCond_ObstrPulmonary"].replace({2: 0}))
beneficiary_clean["ChronicCond_Depression"] = (beneficiary_clean["ChronicCond_Depression"].replace({2: 0}))
beneficiary_clean["ChronicCond_Diabetes"] = (
beneficiary_clean["ChronicCond_Diabetes"].replace({2: 0}))
beneficiary_clean["ChronicCond_IschemicHeart"] = (
beneficiary_clean["ChronicCond_IschemicHeart"].replace({2: 0}))
beneficiary_clean["ChronicCond_Osteoporasis"] = (
beneficiary_clean["ChronicCond_Osteoporasis"].replace({2: 0}))
beneficiary_clean["ChronicCond_rheumatoidarthritis"] = (
beneficiary_clean["ChronicCond_rheumatoidarthritis"].replace({2: 0}))
beneficiary_clean["ChronicCond_stroke"] = (
beneficiary_clean["ChronicCond_stroke"].replace({2: 0}))

print("\nAlzheimer")
print(beneficiary_clean["ChronicCond_Alzheimer"].value_counts(dropna=False))

print("\nHeart Failure")
print(beneficiary_clean["ChronicCond_Heartfailure"].value_counts(dropna=False))

print("\nKidney Disease")
print(beneficiary_clean["ChronicCond_KidneyDisease"].value_counts(dropna=False))

print("\nCancer")
print(beneficiary_clean["ChronicCond_Cancer"].value_counts(dropna=False))

print("\nObstructive Pulmonary")
print(beneficiary_clean["ChronicCond_ObstrPulmonary"].value_counts(dropna=False))

print("\nDepression")
print(beneficiary_clean["ChronicCond_Depression"].value_counts(dropna=False))

print("\nDiabetes")
print(beneficiary_clean["ChronicCond_Diabetes"].value_counts(dropna=False))

print("\nIschemic Heart Disease")
print(beneficiary_clean["ChronicCond_IschemicHeart"].value_counts(dropna=False))

print("\nOsteoporosis")
print(beneficiary_clean["ChronicCond_Osteoporasis"].value_counts(dropna=False))

print("\nRheumatoid Arthritis")
print(beneficiary_clean["ChronicCond_rheumatoidarthritis"].value_counts(dropna=False))

print("\nStroke")
print(beneficiary_clean["ChronicCond_stroke"].value_counts(dropna=False))

## Gender
print("\nGender Values")
print(beneficiary_clean["Gender"].value_counts(dropna=False))

## Race
print("\nRace Value")
print(beneficiary_clean["Race"].value_counts(dropna=False))
print(beneficiary_clean["Race"].isna().sum())

## State
print("\nState Values")
print(beneficiary_clean["State"].value_counts().sort_index())

print("\nState Missing Values")
print(beneficiary_clean["State"].isna().sum())

## County
print("\nCounty Values")
print(beneficiary_clean["County"].value_counts().sort_index())
print(beneficiary_clean["County"].isna().sum())

## Part A Months
print("\nPart A Coverage Months")
print(beneficiary_clean["NoOfMonths_PartACov"].value_counts().sort_index())

## Part B Months
print("\nPart B Coverage Months")
print(beneficiary_clean["NoOfMonths_PartBCov"].value_counts().sort_index())

## Inpatient Annual Reimbursement
print("\nIP Annual Reimbursement")
print(beneficiary_clean["IPAnnualReimbursementAmt"].describe())
print(beneficiary_clean["IPAnnualReimbursementAmt"].isna().sum())
print((beneficiary_clean["IPAnnualReimbursementAmt"] < 0).sum())
print(beneficiary_clean[beneficiary_clean["IPAnnualReimbursementAmt"] < 0][["BeneID", "IPAnnualReimbursementAmt"]].head(20))

## Inpatient Deductible 
print("\nIP Annual Deductible")
print(beneficiary_clean["IPAnnualDeductibleAmt"].describe())

## Outpatient Annual Reimbursement
print("\nOP Annual Reimbursement") 
print(beneficiary_clean["OPAnnualReimbursementAmt"].describe())
print((beneficiary_clean["OPAnnualReimbursementAmt"] < 0).sum())
print(beneficiary_clean[beneficiary_clean["OPAnnualReimbursementAmt"] < 0][["BeneID", "OPAnnualReimbursementAmt"]])

## Outpatient Deductible
print("\nOutpatient Deductible")
print(beneficiary_clean["OPAnnualDeductibleAmt"].describe())

## Provider data
##====================================================================================================

## Encoding PotentialFraud

print(provider_clean["PotentialFraud"].value_counts())

provider_clean["PotentialFraudFlag"] = provider_clean["PotentialFraud"].map({"No": 0, "Yes": 1})

print("\nCounting after encoding")
print(provider_clean[["PotentialFraud", "PotentialFraudFlag"]].head())

print("\nPotential Fraud Flag Missing Values")
print(provider_clean["PotentialFraudFlag"].isna().sum())

print((provider_clean[[
                "total_claims",
                "unique_claims",
                "unique_patients",
                "inpatient_claims",
                "outpatient_claims",
                "total_reimbursement",
                "avg_reimbursement",
                "total_deductible",
                "avg_deductible",
                "avg_claim_duration",
                "claim_per_patient",
                "inpatient_claim_ratio",
                "outpatient_claim_ratio",
                "reimbursement_per_patient",
                "deductible_per_patient"]] < 0).sum())

## Provider Claim Ratio

print("\nInpatient Claim Ratio")
print(((provider_clean["inpatient_claim_ratio"] < 0) | (provider_clean["inpatient_claim_ratio"] > 1)).sum())

print("\nOutpatient Claim Ration")
print(((provider_clean["outpatient_claim_ratio"] < 0) | (provider_clean["outpatient_claim_ratio"] > 1)).sum())

##===================================================================================================
## Save Cleaned Data
##===================================================================================================

provider_clean_file = PROCESSED_FOLDER / "provider_clean.csv"
claims_clean_file = PROCESSED_FOLDER / "claims_clean.csv"
beneficiary_clean_file = PROCESSED_FOLDER / "beneficiary_clean.csv"

provider_clean.to_csv(provider_clean_file,index= False)
claims_clean.to_csv(claims_clean_file,index=False)
beneficiary_clean.to_csv(beneficiary_clean_file,index=False)

print(provider_clean_file.exists())
print(claims_clean_file.exists())
print(beneficiary_clean_file.exists())