##===================================================================================================
## Load Cleaned Data
##===================================================================================================

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed" / "Train"

provider_clean_file = PROCESSED_FOLDER / "provider_clean.csv"
claims_clean_file = PROCESSED_FOLDER / "claims_clean.csv"
beneficiary_clean_file = PROCESSED_FOLDER / "beneficiary_clean.csv"

provider_clean = pd.read_csv(provider_clean_file, low_memory= False)
claims_clean = pd.read_csv(claims_clean_file, low_memory=False, parse_dates= ["ClaimStartDt", "ClaimEndDt"])
beneficiary_clean = pd.read_csv(beneficiary_clean_file, low_memory= False, parse_dates= ["DOB", "DOD"])

print("\nData Shape")
print(provider_clean.shape)
print(claims_clean.shape)
print(beneficiary_clean.shape)

##===================================================================================================
## Merge Claim - Beneficiary Data
##===================================================================================================

print("\nMerge Claim-Beneficiary Data")

claims_beneficiary = claims_clean.merge(beneficiary_clean, on = "BeneID", how = "left", validate = "many_to_one")
print(claims_beneficiary.shape)

print("\nMissing Values")
print(claims_beneficiary["DOB"].isna().sum())
print(claims_beneficiary["Gender"].isna().sum())
print(claims_beneficiary["Race"].isna().sum())

##====================================================================================================
## Save Merged Data
##====================================================================================================

claims_beneficiary_file = PROCESSED_FOLDER / "claim_beneficiary_clean.csv"

claims_beneficiary.to_csv(claims_beneficiary_file, index=False)

print(claims_beneficiary_file.exists())

