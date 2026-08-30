##=============================================
## Creating Path Folder for Accessing Location
##=============================================

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FOLDER = PROJECT_ROOT / "Data" / "Raw" / "Train"

provider_file = TRAIN_FOLDER / "Final_Provider_Train.csv"
claims_file = TRAIN_FOLDER / "All_Claims_Train.csv"
beneficiary_file = TRAIN_FOLDER / "Train_Beneficiarydata-1542865627584.csv"

print("Provider file exists:", provider_file.exists())
print("Claims file exists:", claims_file.exists())
print("Beneficiary file exists:", beneficiary_file.exists())


## Load Provider Data 
##===================

provider_data = pd.read_csv(provider_file, low_memory= False)

print("First 5 rows:")

print(provider_data.head())


## Load Claims Data
##==================

claims_data = pd.read_csv(
    claims_file,
    low_memory=False
)

print("\nFirst five claims:")

print(claims_data.head())


## Load Beneficiary Data
##=======================

beneficiary_data = pd.read_csv(
    beneficiary_file,
    low_memory=False
)

print("\nFirst five beneficiary rows:")

print(beneficiary_data.head())


##====================
## Data Inspection
##=====================


## Provider Data
##================
print("Data Shape")
print(provider_data.shape)

print("\nColumn Names")
print(provider_data.columns.tolist())

print("\nData Types")
print(provider_data.dtypes)

print("\nMissing values")
print(provider_data.isna().sum())

print("\nDuplicate Values")
print(provider_data.duplicated().sum())

print("\nUnique Providers")
print(provider_data["Provider"].nunique())

print("\nPotential Fraud Count")
print(provider_data["PotentialFraud"].value_counts())

## Claims Data
##===============

print("\nData Shape")
print(claims_data.shape)

print("\nCoulmn Names")
print(claims_data.columns.tolist())

print("\nData Types")
print(claims_data.dtypes)

print("\nMissing Values")
print(claims_data.isna().sum())

print("\nDuplicate values")
print(claims_data.duplicated().sum())

print("\nUnique Claims")
print(claims_data["ClaimID"].nunique())

print("\nUnique Providers")
print(claims_data["Provider"].nunique())

print("\nUnique Beneficiary")
print(claims_data["BeneID"].nunique())

print("\nClaim Type")
print(claims_data["claim_type"].value_counts())

## Beneficiary Data
##===================

print("\nData Shape")
print(beneficiary_data.shape)

print("\nColumn Names")
print(beneficiary_data.columns.tolist())

print("\nData Types")
print(beneficiary_data.dtypes)

print("\nMissing Values")
print(beneficiary_data.isna().sum())

print("\nDuplicate values")
print(beneficiary_data.duplicated().sum())

print("\nUnique values")
print(beneficiary_data["BeneID"].nunique())