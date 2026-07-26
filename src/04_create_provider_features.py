##=========================================================================================================================
## Load Data
##=========================================================================================================================

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"

claims_beneficiary_file = PROCESSED_FOLDER / "claim_beneficiary_clean.csv"

claims_beneficiary = pd.read_csv(claims_beneficiary_file, low_memory= False, parse_dates = ["ClaimStartDt", "ClaimEndDt", "DOB", "DOD"])


## Checking Merged Data
##=========================================================================================================================

print(claims_beneficiary.shape)

print(claims_beneficiary[["Provider", "BeneID", "ClaimStartDt", "DOB", "Gender", "Race"]].head())
#
#
#
#
#
#
#
#
#
##=========================================================================================================================
## 1. Demographic Feature Creation
##=========================================================================================================================
#
#
#
#
## Age at Claim
##=========================================================================================================================
claims_beneficiary["age_at_claim"] = ((claims_beneficiary["ClaimStartDt"] - claims_beneficiary["DOB"]).dt.days/365.25).astype(int)   
print("\nAge at Claim")
print(claims_beneficiary["age_at_claim"].describe())


## Provider Beneficiary Table
##=========================================================================================================================

provider_patient = (claims_beneficiary.sort_values(["Provider", "BeneID", "ClaimStartDt"]).drop_duplicates(subset=["Provider", "BeneID"],keep = "first").copy())
print("\nProvider Beneficiary Data")
print(provider_patient.shape)

print("\nDuplicated values")
print(provider_patient.duplicated(subset= ["Provider", "BeneID"]).sum())

print("\nUnique Provider")
print(provider_patient["Provider"].nunique())

print(provider_patient.head())

##========================================================================================================================
## Feature - Age
##========================================================================================================================


## Provider- Avg & Median Beneficiary Age
##========================================================================================================================

provider_demographics = provider_patient.groupby("Provider", as_index=False).agg(unique_patients=("BeneID", "nunique"), avg_patient_age=("age_at_claim", "mean"), median_patient_age= ("age_at_claim", "median"))

provider_demographics["avg_patient_age"] = provider_demographics["avg_patient_age"].round(2)

provider_demographics["median_patient_age"] = provider_demographics["median_patient_age"].round(2)

print("\nProvider Shape")
print(provider_demographics.shape)

print("\nProvider Demographics head")
print(provider_demographics.head())


##========================================================================================================================
## Feature - Gender
##========================================================================================================================


## Provider - Gender Count
##========================================================================================================================
provider_gender = (provider_patient.groupby(["Provider", "Gender"]).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0).reset_index())
provider_gender = provider_gender.rename(columns = {1: "male_patient", 2: "female_patient"})

print("\nProvider Gender Shape")
print(provider_gender.shape)

print("\nProvider Gender Head")
print(provider_gender.head())


## Provider - Gender Percentage
##=========================================================================================================================

provider_gender["total_gender_patients"] = (provider_gender["male_patient"] + provider_gender["female_patient"])
provider_gender["male_patient_percentage"] = ((provider_gender["male_patient"] / provider_gender["total_gender_patients"])*100).round(2)
provider_gender["female_patient_percentage"] = ((provider_gender["female_patient"] / provider_gender["total_gender_patients"])*100).round(2)

print("\nProvider - Gender Percentage")
print(provider_gender.head())


##=========================================================================================================================
## Feature : Race
##=========================================================================================================================


## Provider - Race Count
##=========================================================================================================================

provider_race = (provider_patient.groupby(["Provider", "Race"]).size().unstack(fill_value=0).reindex(columns=[1,2,3,5], fill_value=0).reset_index())
provider_race = provider_race.rename(columns = {1: "race_1", 2: "race_2", 3: "race_3", 5: "race_5"})


## Provicer - Race Percentage
##==========================================================================================================================

provider_race["total_race_patients"] = (provider_race["race_1"] + provider_race["race_2"] + provider_race["race_3"] + provider_race["race_5"])
provider_race["race_1_percentage"] = (provider_race["race_1"] / provider_race["total_race_patients"]*100).round(2)
provider_race["race_2_percentage"] = (provider_race["race_2"] / provider_race["total_race_patients"]*100).round(2)
provider_race["race_3_percentage"] = (provider_race["race_3"] / provider_race["total_race_patients"]*100).round(2)
provider_race["race_5_percentage"] = (provider_race["race_5"] / provider_race["total_race_patients"]*100).round(2)

print("\nProvider - Race Head")
print(provider_race.head())


##==========================================================================================================================
## Merge All 3 Data
##==========================================================================================================================

provider_demographics = provider_demographics.merge(provider_gender, on= "Provider", how = "left", validate = "one_to_one")
provider_demographics = provider_demographics.merge(provider_race, on = "Provider", how = "left", validate = "one_to_one")

print("\nMerged Data Shape")
print(provider_demographics.shape)

print("\nMerged Data Head")
print(provider_demographics.head())

print("\nUnique Provider Check")
print(provider_demographics["Provider"].nunique())   

## Final Data Verification
##=========================================================================================================================

print("\nGender Unique Patients")
print((provider_demographics['unique_patients'] == provider_demographics["total_gender_patients"]).value_counts())

print("\nRace Unique Patients")
print((provider_demographics["unique_patients"] == provider_demographics["total_race_patients"]).value_counts())


## Save Data
##=========================================================================================================================

provider_demographics_file = (PROCESSED_FOLDER / "provider_demographic.csv")

provider_demographics.to_csv(provider_demographics_file, index= False)

print("\n File exist")
print(provider_demographics_file.exists())
#
#
#
#
#
#
#
#
#
#
##=========================================================================================================================
## 2. Chronic Condition Features
##==========================================================================================================================


##=========================================================================================================================
## Feature: Chronic Condition
##=========================================================================================================================


## Chronic Condition Count
##=========================================================================================================================

provider_patient["chronic_condition_count"] = (provider_patient["ChronicCond_Alzheimer"] + provider_patient["ChronicCond_Heartfailure"] + provider_patient["ChronicCond_KidneyDisease"] + provider_patient["ChronicCond_Cancer"] + provider_patient["ChronicCond_ObstrPulmonary"] + provider_patient["ChronicCond_Depression"]  + provider_patient["ChronicCond_Diabetes"] + provider_patient["ChronicCond_IschemicHeart"] + provider_patient["ChronicCond_Osteoporasis"] + provider_patient["ChronicCond_rheumatoidarthritis"] + provider_patient["ChronicCond_stroke"])

print("\nChronic Condition Count")
print(provider_patient["chronic_condition_count"].describe())


## Chronic Condition Avg & Median
##==========================================================================================================================

provider_chronic = (provider_patient.groupby("Provider", as_index= False).agg(avg_chronic_condition_count=("chronic_condition_count", "mean"), median_chronic_condition_count =("chronic_condition_count", "median")))

provider_chronic["avg_chronic_condition_count"] = (provider_chronic["avg_chronic_condition_count"].round(2))
provider_chronic["median_chronic_condition_count"] = (provider_chronic["median_chronic_condition_count"].round(2))

print("\nProvider Chronic Shape")
print(provider_chronic.shape)

print("\nProvider Chronic Head")
print(provider_chronic.head())


## Provider Disease Count
##=========================================================================================================================

provider_disease_counts = provider_patient.groupby("Provider", as_index=False).agg(alzheimer_patient_count=("ChronicCond_Alzheimer", "sum"), heartfailure_patient_count=("ChronicCond_Heartfailure", "sum"), kidney_disease_patient_count=("ChronicCond_KidneyDisease", "sum"), cancer_patient_count=("ChronicCond_Cancer", "sum"), obstructive_pulmonary_patient_count=("ChronicCond_ObstrPulmonary", "sum"), depression_patient_count=("ChronicCond_Depression", "sum"), diabetes_patient_count=("ChronicCond_Diabetes", "sum"), ischemic_heart_patient_count=("ChronicCond_IschemicHeart", "sum"), osteoporosis_patient_count=("ChronicCond_Osteoporasis", "sum"), rheumatoid_arthritis_patient_count=("ChronicCond_rheumatoidarthritis", "sum"), stroke_patient_count=("ChronicCond_stroke", "sum"), renal_disease_patient_count=("RenalDiseaseIndicator", "sum"))

print("\nProvider Diease Shape")
print(provider_disease_counts.shape)

print("\nProvider Dieases Head")
print(provider_disease_counts.head())

print("\nMissing Values")
print(provider_disease_counts.isna().sum())


## Provider Disease Rate
##=========================================================================================================================

provider_disease_rates = provider_patient.groupby("Provider")[["ChronicCond_Alzheimer","ChronicCond_Heartfailure","ChronicCond_KidneyDisease","ChronicCond_Cancer","ChronicCond_ObstrPulmonary","ChronicCond_Depression","ChronicCond_Diabetes","ChronicCond_IschemicHeart","ChronicCond_Osteoporasis","ChronicCond_rheumatoidarthritis","ChronicCond_stroke","RenalDiseaseIndicator"]].mean().mul(100).round(2).reset_index().rename(columns={"ChronicCond_Alzheimer":"alzheimer_patient_rate","ChronicCond_Heartfailure":"heartfailure_patient_rate","ChronicCond_KidneyDisease":"kidney_disease_patient_rate","ChronicCond_Cancer":"cancer_patient_rate","ChronicCond_ObstrPulmonary":"obstructive_pulmonary_patient_rate","ChronicCond_Depression":"depression_patient_rate","ChronicCond_Diabetes":"diabetes_patient_rate","ChronicCond_IschemicHeart":"ischemic_heart_patient_rate","ChronicCond_Osteoporasis":"osteoporosis_patient_rate","ChronicCond_rheumatoidarthritis":"rheumatoid_arthritis_patient_rate","ChronicCond_stroke":"stroke_patient_rate","RenalDiseaseIndicator":"renal_disease_patient_rate"})

print("\nProvider Disease Rate Shape")
print(provider_disease_rates.shape)

print("\nProvider Disease Rate Head")
print(provider_disease_rates.head())


##Merge Chronic Disease Data
##=========================================================================================================================

provider_chronic_features = provider_chronic.merge( provider_disease_counts, on = "Provider", how = "left", validate = "one_to_one")
provider_chronic_features = provider_chronic_features.merge(provider_disease_rates, on = "Provider", how= "left", validate = "one_to_one")

print("\nChronic Features Shape")
print(provider_chronic_features.shape)

print("\nChronic Features Head")
print(provider_chronic_features.head())


## Save Data
##=========================================================================================================================

provider_chronic_features_file = PROCESSED_FOLDER / "provider_chronic_features.csv"

provider_chronic_features.to_csv(provider_chronic_features_file, index = False)

print(provider_chronic_features_file.exists())