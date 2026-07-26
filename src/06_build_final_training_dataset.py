##====================================================================================================
## Load Data
##=======================================================================================================


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"

provider_clean_file = (PROCESSED_FOLDER / "provider_clean.csv")

provider_demographic_file = (PROCESSED_FOLDER / "provider_demographic.csv")

provider_chronic_features_file = (PROCESSED_FOLDER / "provider_chronic_features.csv")

provider_fraud_risk_features_file = (PROCESSED_FOLDER / "provider_fraud_risk_features.csv")

print(provider_clean_file.exists())
print(provider_demographic_file.exists())
print(provider_chronic_features_file.exists())
print(provider_fraud_risk_features_file.exists())


## Load All Data Files
##=====================================================================================================

provider_clean = pd.read_csv(provider_clean_file, low_memory=False)
provider_demographic = pd.read_csv(provider_demographic_file, low_memory=False)
provider_chronic_features = pd.read_csv(provider_chronic_features_file, low_memory=False)
provider_fraud_risk_features = pd.read_csv(provider_fraud_risk_features_file, low_memory=False)

##========================================================================================================
## Provider All Table Validation
##=========================================================================================================

print("\nProvider Clean Shape")
print(provider_clean.shape)

print("\nProvider Demographics Shape")
print(provider_demographic.shape)

print("\nProvider Chronic Features Shape")
print(provider_chronic_features.shape)

print("\nProvider Fraud Risk Features Shape")
print(provider_fraud_risk_features.shape)



print("\nDuplicate Providers in Provider Clean")
print(provider_clean["Provider"].duplicated().sum())

print("\nDuplicate Providers in Demographics")
print(provider_demographic["Provider"].duplicated().sum())

print("\nDuplicate Providers in Chronic Features")
print(provider_chronic_features["Provider"].duplicated().sum())

print("\nDuplicate Providers in Fraud Risk Features")
print(provider_fraud_risk_features["Provider"].duplicated().sum())


##=======================================================================================================
## Check Overlapping Feature Columns
##========================================================================================================

provider_clean_columns = (set(provider_clean.columns)- {"Provider"})

provider_demographic_columns = (set(provider_demographic.columns)- {"Provider"})

provider_chronic_columns = (set(provider_chronic_features.columns)- {"Provider"})

provider_fraud_risk_columns = (set(provider_fraud_risk_features.columns)- {"Provider"})


print("\nProvider Clean and Demographics Overlap")
print(sorted(provider_clean_columns.intersection(provider_demographic_columns)))

print("\nProvider Clean and Chronic Features Overlap")
print(sorted(provider_clean_columns.intersection(provider_chronic_columns)))

print("\nProvider Clean and Fraud Risk Features Overlap")
print(sorted(provider_clean_columns.intersection(provider_fraud_risk_columns)))

print("\nDemographics and Chronic Features Overlap")
print(sorted(provider_demographic_columns.intersection(provider_chronic_columns)))

print("\nDemographics and Fraud Risk Features Overlap")
print(sorted(provider_demographic_columns.intersection(provider_fraud_risk_columns)))

print("\nChronic and Fraud Risk Features Overlap")
print(sorted(provider_chronic_columns.intersection(provider_fraud_risk_columns)))


##====================================================================================================
## Dropping Unique Patient Columns
##====================================================================================================

## Verifying Unique Patient Column in both Table
##====================================================================================================


unique_patient_check = provider_clean[["Provider", "unique_patients"]].merge(provider_demographic[["Provider", "unique_patients"]], on= "Provider", how = "inner", validate= "one_to_one", suffixes=("_provider_clean", "_demographic"))

print((unique_patient_check["unique_patients_provider_clean"] != unique_patient_check["unique_patients_demographic"]).sum())


## Drop Duplicate Unique Patients Column
##=====================================================================================================

provider_demographic_for_merge = provider_demographic.drop(columns =["unique_patients"])

##=====================================================================================================
## Merge All Data
##=====================================================================================================`    `

final_training_data = provider_clean.merge(provider_demographic_for_merge, on = "Provider", how = "left", validate= "one_to_one")

final_training_data = final_training_data.merge(provider_chronic_features, on= "Provider", how= "left", validate="many_to_one")

final_training_data = final_training_data.merge(provider_fraud_risk_features, on= "Provider", how = "left", validate= "one_to_one")

print("\nFinal Data Shape")
print(final_training_data.shape)

print("\nFinal Data Head")
print(final_training_data.head())


print("\nPotential Fraud Values")
print(final_training_data["PotentialFraud"].value_counts())

print("\nPotential Fraud Flag Values")
print(final_training_data["PotentialFraudFlag"].value_counts())


## Save Data
##===================

final_training_data_file = PROCESSED_FOLDER / "final_training_dataset.csv"

final_training_data.to_csv(final_training_data_file, index=False)

print(final_training_data_file.exists())
