from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"
OUTPUT_FOLDER = PROJECT_ROOT / "output" / "models"
MODELS_FOLDER = OUTPUT_FOLDER / "models"

MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

final_training_data_file = PROCESSED_FOLDER / "final_training_dataset.csv"

final_training_data = pd.read_csv(final_training_data_file, low_memory=False)

print(final_training_data.shape)

print(final_training_data.columns.to_list)


##=========================================================================================================================
## X and Y  Variable
##=========================================================================================================================

y = final_training_data["PotentialFraudFlag"]

print("\nY shape")
print(y.shape)

exclude_columns =  ["Provider",
    "PotentialFraud",
    "PotentialFraudFlag",
    "unique_claims",
    "inpatient_claims",
    "outpatient_claims",
    "outpatient_claim_ratio",
    "male_patient",
    "female_patient",
    "total_gender_patients",
    "female_patient_percentage",
    "race_1",
    "race_2",
    "race_3",
    "race_5",
    "total_race_patients",
    "median_patient_age",
    "median_chronic_condition_count",
    "alzheimer_patient_count",
    "heartfailure_patient_count",
    "kidney_disease_patient_count",
    "cancer_patient_count",
    "obstructive_pulmonary_patient_count",
    "depression_patient_count",
    "diabetes_patient_count",
    "ischemic_heart_patient_count",
    "osteoporosis_patient_count",
    "rheumatoid_arthritis_patient_count",
    "stroke_patient_count",
    "renal_disease_patient_count",
    "high_cost_claim_count"]

x = final_training_data.drop(columns= exclude_columns)

print("\nX shape")
print(x.shape)

print(x.isna().sum())

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state = 42, stratify=y)

print("\nX train shape")
print(x_train.shape)

print("\nX test shape")
print(x_test.shape)

print("\nY dist train")
print(y_train.value_counts().to_string())

print("\nY dist test")
print(y_test.value_counts().to_string())


##=========================================================================================================================
## Save Models
##=========================================================================================================================

x_train_file = MODELS_FOLDER / "x_train.csv"
x_test_file = MODELS_FOLDER / "x_test.csv"
y_train_file = MODELS_FOLDER / "y_train.csv"
y_test_file = MODELS_FOLDER / "y_test.csv"

x_train.to_csv(x_train_file, index = False)
x_test.to_csv(x_test_file, index = False)
y_train.to_csv(y_train_file, index = False)
y_test.to_csv(y_test_file, index = False)

print(x_train_file)
print(x_test_file)
print(y_train_file)
print(y_test_file)