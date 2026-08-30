from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FOLDER = PROJECT_ROOT / "Data" / "Raw" / "Test"

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed" / "Test"

provider_test_file = TEST_FOLDER / "Final_Provider_Test.csv"
claims_test_file = TEST_FOLDER / "All_Claims_Test.csv"
beneficiary_test_file = TEST_FOLDER / "Test_Beneficiarydata-1542969243754.csv"

provider_test = pd.read_csv(provider_test_file, low_memory=False)
claims_test = pd.read_csv (claims_test_file, low_memory=False)
beneficiary_test = pd.read_csv(beneficiary_test_file, low_memory=False)

print("\nData Shape")
print(provider_test.shape)
print(claims_test.shape)
print(beneficiary_test.shape)

print("\nColumn Names")
print(provider_test.columns.to_list())
print(claims_test.columns.to_list())
print(beneficiary_test.columns.tolist())

provider_test = provider_test.rename(columns={"claims_per_patient": "claim_per_patient", "dedcutible_per_patient": "deductible_per_patient"})
print(provider_test.columns.to_list())

##=========================================================================================================================
## Checking Data- Duplicate and Missing Values
##=========================================================================================================================

print("\nProvider Duplicates")
print(provider_test.duplicated().sum())

print("\nClaims Duplicates")
print(claims_test.duplicated().sum())

print("\nBeneficiary Duplicates")
print(beneficiary_test.duplicated().sum())

print("\nProvider Missing")
print(provider_test.isna().sum().to_string())

print("\nClaims Missing")
print(claims_test.isna().sum().to_string())

print("\nBeneficiary Missing")
print(beneficiary_test.isna().sum().to_string())

##=========================================================================================================================
## Clean Test Claim Data
##=========================================================================================================================

claims_test["ClaimStartDt"] = pd.to_datetime(claims_test["ClaimStartDt"], errors="coerce")

claims_test["ClaimEndDt"] = pd.to_datetime(claims_test["ClaimEndDt"], errors = "coerce")

claims_test["DeductibleAmtPaid_missing"] = (claims_test["DeductibleAmtPaid"].isna().astype(int))

missing_by_claim_type = (claims_test[claims_test["DeductibleAmtPaid"].isna()]["claim_type"].value_counts())
print(missing_by_claim_type.to_string())

## Replace Missing values with median values

deductible_median = (claims_test.loc[claims_test["claim_type"] == "Inpatient", "DeductibleAmtPaid"].median())

claims_test["DeductibleAmtPaid"] = (claims_test["DeductibleAmtPaid"].fillna(deductible_median))

print("\nEnd Date-Start Date")
print((claims_test["ClaimEndDt"] < claims_test["ClaimStartDt"]).sum())

print("\nNegative Reimbursement")
print((claims_test["InscClaimAmtReimbursed"] < 0).sum())

print("\nNegative Deductible")
print((claims_test["DeductibleAmtPaid"] < 0).sum())

print(claims_test.shape)

##========================================================================================================================
## Save Claims clean data
##========================================================================================================================

claims_test_clean_file = PROCESSED_FOLDER / "claims_test_clean.csv"

claims_test.to_csv(claims_test_clean_file, index=False)

print(claims_test_clean_file)


##==========================================================================================================================
## Clean Test Beneficiary Data
##==========================================================================================================================

beneficiary_test["DOB"] = pd.to_datetime(beneficiary_test["DOB"], errors= "coerce")

beneficiary_test["DOD"] = pd.to_datetime(beneficiary_test["DOD"], errors= "coerce")

## Encode Chronic Disease Columns

beneficiary_test["ChronicCond_Alzheimer"] = beneficiary_test["ChronicCond_Alzheimer"].replace({2: 0})
beneficiary_test["ChronicCond_Heartfailure"] = beneficiary_test["ChronicCond_Heartfailure"].replace({2: 0})
beneficiary_test["ChronicCond_KidneyDisease"] = beneficiary_test["ChronicCond_KidneyDisease"].replace({2: 0})
beneficiary_test["ChronicCond_Cancer"] = beneficiary_test["ChronicCond_Cancer"].replace({2: 0})
beneficiary_test["ChronicCond_ObstrPulmonary"] = beneficiary_test["ChronicCond_ObstrPulmonary"].replace({2: 0})
beneficiary_test["ChronicCond_Depression"] = beneficiary_test["ChronicCond_Depression"].replace({2: 0})
beneficiary_test["ChronicCond_Diabetes"] = beneficiary_test["ChronicCond_Diabetes"].replace({2: 0})
beneficiary_test["ChronicCond_IschemicHeart"] = beneficiary_test["ChronicCond_IschemicHeart"].replace({2: 0})
beneficiary_test["ChronicCond_Osteoporasis"] = beneficiary_test["ChronicCond_Osteoporasis"].replace({2: 0})
beneficiary_test["ChronicCond_rheumatoidarthritis"] = beneficiary_test["ChronicCond_rheumatoidarthritis"].replace({2: 0})
beneficiary_test["ChronicCond_stroke"] = beneficiary_test["ChronicCond_stroke"].replace({2: 0})

## Renal Disease Encoding

beneficiary_test["RenalDiseaseIndicator"] = beneficiary_test["RenalDiseaseIndicator"].replace({"Y": 1, "0": 0, 0:0})
beneficiary_test["RenalDiseaseIndicator"] = beneficiary_test["RenalDiseaseIndicator"].astype(int)

## Missing values

print("\nMissing Values- Beneficiary")
print(beneficiary_test.isna().sum().to_string())

print("\nDOD < DOB")
print((beneficiary_test["DOD"] < beneficiary_test["DOB"]).sum())

print(beneficiary_test["Gender"].value_counts().sort_index())

print(beneficiary_test["Race"].value_counts().sort_index())

print(beneficiary_test["RenalDiseaseIndicator"].value_counts().sort_index())

print(beneficiary_test.shape)

##==========================================================================================================================
## Save Beneficiary Data and Provider data
##==========================================================================================================================

beneficiary_test_clean_file = PROCESSED_FOLDER / "beneficiary_test_clean.csv"

beneficiary_test.to_csv(beneficiary_test_clean_file, index=False)

print(beneficiary_test_clean_file)

provider_test_clean_file = PROCESSED_FOLDER / "provider_test.csv"

provider_test.to_csv(provider_test_clean_file, index = False)

print(provider_test_clean_file)


##==========================================================================================================================
## Merge Claims and Beneficiary Data
##==========================================================================================================================

claim_beneficiary_test = claims_test.merge(beneficiary_test, on= "BeneID", how= "left", validate= "many_to_one")

print("\nClaim Beneficiary Shape")
print(claim_beneficiary_test.shape)

print("\nMissing values after merge")
print(claim_beneficiary_test[["DOB", "Gender", "Race", "RenalDiseaseIndicator"]].isna().sum().to_string())

print("\nClaim ID unique before vs after")
print(claims_test["ClaimID"].nunique())
print(claim_beneficiary_test["ClaimID"].nunique())

print("\nRows before vs After")
print(claims_test.shape)
print(claim_beneficiary_test.shape)

##========================================================================================================================
## Save Claim & Beneficiary merged Data
##========================================================================================================================

claim_beneficiary_test_file = PROCESSED_FOLDER / "claim_beneficiary_test.csv"

claim_beneficiary_test.to_csv(claim_beneficiary_test_file, index=False)
print(claim_beneficiary_test_file)

#
#
#
#
#
##========================================================================================================================
## 1) Creating Demographic Features
##========================================================================================================================


## Age at Claim
##========================================================================================================================

claim_beneficiary_test["age_at_claim"] = ((claim_beneficiary_test["ClaimStartDt"] - claim_beneficiary_test["DOB"]).dt.days / 365.25).astype(int)

## Patient Per Provider
##========================================================================================================================

provider_patient_test = (claim_beneficiary_test.sort_values(["Provider", "BeneID", "ClaimStartDt"]).drop_duplicates(subset=["Provider", "BeneID"], keep= "first").copy())

print("\nPatient Per Provider data")
print(provider_patient_test.shape)

## Age Features
##========================================================================================================================

age_features_test = (provider_patient_test.groupby("Provider").agg(unique_patients= ("BeneID", "nunique"), avg_patient_age= ("age_at_claim", "mean"), median_patient_age= ("age_at_claim", 'median')).reset_index())

print("\nAge Features")
print(age_features_test.shape)

## Gender Features
##========================================================================================================================

gender_features_test = (provider_patient_test.groupby(["Provider", "Gender"]).size().unstack(fill_value=0))

gender_features_test = gender_features_test.reindex(columns=[1,2], fill_value=0)

gender_features_test = gender_features_test.rename(columns={1: "male_patient", 2:"female_patient"})

gender_features_test = gender_features_test.reset_index()

gender_features_test["total_gender_patients"] = (gender_features_test["male_patient"] + gender_features_test["female_patient"])

gender_features_test["male_patient_percentage"] = (gender_features_test["male_patient"] / gender_features_test["total_gender_patients"])*100

gender_features_test["female_patient_percentage"] = (gender_features_test["female_patient"]/ gender_features_test["total_gender_patients"])*100

print("\nGender Features")
print(gender_features_test.shape)

## Race Features
##=========================================================================================================================

race_features_test= (provider_patient_test.groupby(["Provider", "Race"]).size().unstack(fill_value=0))

race_features_test = race_features_test.reindex(columns=[1,2,3,5], fill_value=0)

race_features_test = race_features_test.rename(columns={1:"race_1", 2: "race_2", 3: "race_3", 5: "race_5"})

race_features_test = race_features_test.reset_index()

race_features_test["total_race_patient"] = (race_features_test["race_1"] + race_features_test["race_2"] + race_features_test["race_3"]+ race_features_test["race_5"])

race_features_test["race_1_percentage"] = (race_features_test["race_1"] / race_features_test["total_race_patient"]) * 100

race_features_test["race_2_percentage"] = (race_features_test["race_2"] / race_features_test["total_race_patient"]) * 100

race_features_test["race_3_percentage"] = (race_features_test["race_3"] / race_features_test["total_race_patient"]) * 100

race_features_test["race_5_percentage"] = (race_features_test["race_5"] / race_features_test["total_race_patient"]) * 100

print("\nRace Features")
print(race_features_test.shape)


## Merge All Features 
##========================================================================================================================

provider_demographic_test = age_features_test.merge(gender_features_test, on = "Provider", how= "left", validate= "one_to_one")

provider_demographic_test = provider_demographic_test.merge(race_features_test, on = "Provider", how = "left", validate= "one_to_one")

print("\nProvider Demographic Test Shape")
print(provider_demographic_test.shape)

print(provider_demographic_test.isna().sum().to_string())

##=========================================================================================================================
## Save Provider Demographic Data
##=========================================================================================================================

provider_demographic_test_file = PROCESSED_FOLDER / "provider_demographic_test.csv"

provider_demographic_test.to_csv(provider_demographic_test_file, index=False)

print(provider_demographic_test_file)


##=========================================================================================================================
## 2) Chronic Condition Features
##=========================================================================================================================

provider_patient_test["chronic_condition_count"] = (provider_patient_test["ChronicCond_Alzheimer"] + provider_patient_test["ChronicCond_Heartfailure"] + provider_patient_test["ChronicCond_KidneyDisease"]+ provider_patient_test["ChronicCond_Cancer"]+ provider_patient_test["ChronicCond_ObstrPulmonary"] + provider_patient_test["ChronicCond_Depression"]+ provider_patient_test["ChronicCond_Depression"]+ provider_patient_test["ChronicCond_Diabetes"] + provider_patient_test["ChronicCond_IschemicHeart"]+ provider_patient_test["ChronicCond_rheumatoidarthritis"] + provider_patient_test["ChronicCond_stroke"])

provider_chronic_test = (provider_patient_test.groupby("Provider").agg(unique_patients =("BeneID", "nunique"), avg_chronic_condition_count = ("chronic_condition_count", "mean"), median_chronic_condition_count = ("chronic_condition_count", "median"),   alzheimer_patient_count=("ChronicCond_Alzheimer", "sum"),
        heartfailure_patient_count=("ChronicCond_Heartfailure", "sum"),
        kidney_disease_patient_count=("ChronicCond_KidneyDisease", "sum"),
        cancer_patient_count=("ChronicCond_Cancer", "sum"),
        obstructive_pulmonary_patient_count=("ChronicCond_ObstrPulmonary", "sum"),
        depression_patient_count=("ChronicCond_Depression", "sum"),
        diabetes_patient_count=("ChronicCond_Diabetes", "sum"),
        ischemic_heart_patient_count=("ChronicCond_IschemicHeart", "sum"),
        osteoporosis_patient_count=("ChronicCond_Osteoporasis", "sum"),
        rheumatoid_arthritis_patient_count=("ChronicCond_rheumatoidarthritis", "sum"),
        stroke_patient_count=("ChronicCond_stroke", "sum"),
        renal_disease_patient_count=("RenalDiseaseIndicator", "sum")).reset_index())


## Chronic Disease Rate
##=========================================================================================================================

provider_disease_rates_test = (provider_patient_test.groupby("Provider")[["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", "RenalDiseaseIndicator"]].mean().mul(100).round(2).reset_index().rename(columns={"ChronicCond_Alzheimer" : "alzheimer_patient_rate", "ChronicCond_Heartfailure": "heartfailure_patient_rate","ChronicCond_KidneyDisease": "kidney_disease_patient_rate","ChronicCond_Cancer": "cancer_patient_rate","ChronicCond_ObstrPulmonary": "obstructive_pulmonary_patient_rate","ChronicCond_Depression": "depression_patient_rate","ChronicCond_Diabetes": "diabetes_patient_rate","ChronicCond_IschemicHeart": "ischemic_heart_patient_rate","ChronicCond_Osteoporasis": "osteoporosis_patient_rate","ChronicCond_rheumatoidarthritis": "rheumatoid_arthritis_patient_rate", "ChronicCond_stroke": "stroke_patient_rate","RenalDiseaseIndicator": "renal_disease_patient_rate" }))

print("\nProvider Disease Rates Shape") 
print(provider_disease_rates_test.shape)

provider_chronic_test = provider_chronic_test.drop(columns=["unique_patients"])

print("\nChronic Test Shape")
print(provider_chronic_test.shape)

print("\nMissing Values")
print(provider_chronic_test.isna().sum().to_string())

## Save Chronic Test Data
##=========================================================================================================================
provider_chronic_test_file = PROCESSED_FOLDER / "provider_chronic_test.csv"

provider_chronic_test.to_csv(provider_chronic_test_file, index=False)

print(provider_chronic_test_file)


##=========================================================================================================================
## Merge and Save Chronic Count and Rate
##=========================================================================================================================

provider_chronic_test = provider_chronic_test.merge(provider_disease_rates_test, on= "Provider", how= "left", validate="one_to_one")

print("\nShape")
print(provider_chronic_test.shape)
print(provider_chronic_test.isna().sum().to_string())

print(provider_chronic_test.columns.tolist())

provider_chronic_test_file = PROCESSED_FOLDER / "provider_chronic_test.csv"


provider_chronic_test.to_csv(provider_chronic_test_file, index=False)

print(provider_chronic_test_file)


##=========================================================================================================================
## 3) Fraud Risk Feature Creation
##=========================================================================================================================

claim_beneficiary_test["claim_duration_days"] = (claim_beneficiary_test["ClaimEndDt"] - claim_beneficiary_test["ClaimStartDt"]).dt.days

## Using same training data threshold
##=========================================================================================================================
PROCESSED_FOLDER_TRAIN = PROJECT_ROOT / "Data" / "Processed" / "Train"

training_claims_file = PROCESSED_FOLDER_TRAIN / "claims_clean.csv"

training_claims = pd.read_csv(training_claims_file, low_memory=False)

high_cost_threshold = (training_claims.groupby("claim_type")["InscClaimAmtReimbursed"].quantile(0.95))

print(high_cost_threshold)

## Applying it to Test data

claim_beneficiary_test["high_cost_threshold"] = (claim_beneficiary_test["claim_type"].map(high_cost_threshold))

claim_beneficiary_test["high_cost_claim_flag"] = (claim_beneficiary_test["InscClaimAmtReimbursed"] >= claim_beneficiary_test["high_cost_threshold"]).astype(int)

print("\nFlag Counts")
print(claim_beneficiary_test["high_cost_claim_flag"].value_counts())


## Provider Fraud Risk Test Features
##=========================================================================================================================

provider_fraud_risk_test = (claim_beneficiary_test.groupby("Provider").agg(median_claim_duration_days = ("claim_duration_days", "median"), max_claim_duration_days = ("claim_duration_days", "max"), total_claims_for_high_cost = ("ClaimID", "count"), high_cost_claim_count= ("high_cost_claim_flag", "sum")).reset_index())


provider_fraud_risk_test["high_cost_claim_percentage"] = (provider_fraud_risk_test["high_cost_claim_count"] / provider_fraud_risk_test["total_claims_for_high_cost"])*100

provider_fraud_risk_test = provider_fraud_risk_test.drop(columns = ["total_claims_for_high_cost"])

print("\nFraud Risk Shape")
print(provider_fraud_risk_test.shape)

print("\nMissing Values")
print(provider_fraud_risk_test.isna().sum().to_string())

provider_fraud_risk_test_file = PROCESSED_FOLDER / "provider_fraud_risk_test.csv"

provider_fraud_risk_test.to_csv(provider_fraud_risk_test_file, index=False)

print(provider_fraud_risk_test_file)

#
#
#
#
##========================================================================================================================
## Final Data set
##========================================================================================================================

print(provider_test.shape)
print(provider_demographic_test.shape)
print(provider_chronic_test.shape)
print(provider_fraud_risk_test.shape)


## Check Overlapping Columns Before Final Test Merge
##=========================================================================================================================

provider_test_columns = set(provider_test.columns)

provider_demographic_test_columns = set(provider_demographic_test.columns)

provider_chronic_test_columns = set(provider_chronic_test.columns)

provider_fraud_risk_test_columns = set(provider_fraud_risk_test.columns)


print("\nProvider Test and Demographics Test Overlap")
print(sorted(provider_test_columns.intersection(provider_demographic_test_columns)))

print("\nProvider Test and Chronic Test Overlap")
print(sorted(provider_test_columns.intersection(provider_chronic_test_columns)))

print("\nProvider Test and Fraud Risk Test Overlap")
print(sorted(provider_test_columns.intersection(provider_fraud_risk_test_columns)))

print("\nDemographics Test and Chronic Test Overlap")
print(sorted(provider_demographic_test_columns.intersection(provider_chronic_test_columns)))

print("\nDemographics Test and Fraud Risk Test Overlap")
print(sorted(provider_demographic_test_columns.intersection(provider_fraud_risk_test_columns)))

print("\nChronic Test and Fraud Risk Test Overlap")
print(sorted(provider_chronic_test_columns.intersection(provider_fraud_risk_test_columns)))

## Check Unique Patients
##=========================================================================================================================

unique_patient_check_test = provider_test[["Provider", "unique_patients"]].merge(provider_demographic_test[["Provider", "unique_patients"]], on="Provider", how="left", suffixes= ("_provider", "_demographic"), validate= "one_to_one")

unique_patient_mismatch_test = unique_patient_check_test["unique_patients_provider"] != unique_patient_check_test["unique_patients_demographic"].sum()

print(unique_patient_mismatch_test.sum())

provider_demographic_test = provider_demographic_test.drop(columns=["unique_patients"])

## Merge
##==========================================================================================================================

final_test_dataset = provider_test.merge(provider_demographic_test, on = "Provider", how= "left", validate= "one_to_one")

final_test_dataset= final_test_dataset.merge(provider_chronic_test, on= "Provider", how= "left", validate= "one_to_one")

final_test_dataset = final_test_dataset.merge(provider_fraud_risk_test, on="Provider", how="left", validate="one_to_one")

print("\nShape and Missing Values")
print(final_test_dataset.shape)
print(final_test_dataset.isna().sum().to_string())

final_test_dataset_file = PROCESSED_FOLDER / "final_test_dataset.csv"

final_test_dataset.to_csv(final_test_dataset_file, index= False)

print(final_test_dataset_file)
#
#
#
#
#
##=========================================================================================================================
## Check Training vs Test Data
##========================================================================================================================

final_training_dataset_file = PROCESSED_FOLDER_TRAIN / "final_training_dataset.csv"

final_training_dataset = pd.read_csv(final_training_dataset_file, low_memory=False)


training_columns = set (final_training_dataset.columns)
test_columns = set(final_test_dataset.columns)

columns_in_training_na_test = sorted(training_columns - test_columns)
columns_in_test_na_training = sorted(test_columns - training_columns)

print("\nColumns in each train and test dataset")
print(columns_in_training_na_test)
print(columns_in_test_na_training)

print(final_training_dataset.shape)
print(final_test_dataset.shape)
final_test_dataset = final_test_dataset.rename(columns={"total_race_patient": "total_race_patients"})

final_test_dataset_file = PROCESSED_FOLDER / "final_test_dataset.csv"

final_test_dataset.to_csv(final_test_dataset_file,index=False)

print("\nCorrected Final Test Dataset Saved")
print(final_test_dataset_file)

print("\nFinal Test Dataset Shape")
print(final_test_dataset.shape)
#
#
#
#
##=========================================================================================================================
## Final test model input
##=========================================================================================================================

x_train_file = PROJECT_ROOT / "output" / "models" / "x_train.csv"

x_train = pd.read_csv(x_train_file, low_memory= False)

x_train_columns = x_train.columns.tolist()

X_test_actual = final_test_dataset[x_train_columns].copy()

print("\nTrain Shape")
print(x_train.shape)

print("\nActual Test Model Input Shape")
print(X_test_actual.shape)

print("\nMissing Values in Actual Test Model")
print(X_test_actual.isna().sum().to_string())

actual_test_file = PROJECT_ROOT / "output" / "models" / "X_test_actual.csv"

X_test_actual.to_csv(actual_test_file, index=False)

print(actual_test_file)
