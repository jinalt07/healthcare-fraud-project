from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"
PROCESSED_FOLDER_TEST = PROJECT_ROOT / "Data" / "Processed" / "Test"
OUTPUT_FOLDER = PROJECT_ROOT / "output"
MODELS_FOLDER = OUTPUT_FOLDER / "models" / "models"

final_test_dataset_file = PROCESSED_FOLDER_TEST / "final_test_dataset.csv"
actual_test_input_file = MODELS_FOLDER / "X_test_actual.csv"
selected_model_file = MODELS_FOLDER / "gradient_boosting_model.pkl"

## Load all files

final_test_dataset = pd.read_csv(final_test_dataset_file, low_memory=False)

X_test_actual = pd.read_csv(actual_test_input_file, low_memory=False)

selected_model= joblib.load(selected_model_file)

print("\nShape")

print(final_test_dataset.shape)

print(X_test_actual.shape)

print("\nModel")

print(selected_model)

## Predicting Fraud Labels & Probabilities on X test
##=========================================================================================================================

predict_fraud_label = selected_model.predict(X_test_actual)

predict_fraud_prob = selected_model.predict_proba(X_test_actual)[:,1]

print("\nFraud Label Count")
print(pd.Series(predict_fraud_label).value_counts())

print("\nFraud Probability")
print(pd.Series(predict_fraud_prob).describe())

## Prediction Table
##=========================================================================================================================

test_prediction = final_test_dataset[["Provider"]].copy()

test_prediction["predict_fraud_label"] = predict_fraud_label

test_prediction["predict_fraud_prob"] = predict_fraud_prob

test_prediction["predict_fraud_status"] = test_prediction["predict_fraud_label"].map({0:"Not Potential Fraud", 1:"Potential Fraud"})

print("\nShape")
print(test_prediction.shape)

print("\nCounts")
print(test_prediction["predict_fraud_status"].value_counts())

print("\nTop 10")
print(test_prediction.sort_values(by="predict_fraud_prob", ascending= False).head(10))

## Save data output
##=========================================================================================================================

test_prediction_file = OUTPUT_FOLDER / "test_provider_predictions.csv"

test_prediction.to_csv(test_prediction_file, index= False)

print(test_prediction_file)

##=========================================================================================================================
## Fruad Risk Level Prediction
##=========================================================================================================================

OUTPUT_FOLDER_2 = PROJECT_ROOT / "output" / "models"
risk_threshold_file = OUTPUT_FOLDER_2 / "risk_threshold.csv"

risk_threshold = pd.read_csv(risk_threshold_file, low_memory=False)

print("\nRisk Threshold")
print(risk_threshold)
print(risk_threshold.columns.tolist())

## Create risk flag and levels
##=========================================================================================================================

## Threshold to Dictionary
risk_threshold_dict = dict(zip(risk_threshold["risk_feature"], risk_threshold["threshold_value"]))

print("\nThreshold Dictionary")
print(risk_threshold_dict)

## Create Flags
final_test_dataset["high_total_claim_flag"] = (final_test_dataset["total_claims"] >= risk_threshold_dict["total_claims"]).astype(int)

final_test_dataset["high_total_reimbursement_flag"] =(final_test_dataset["total_reimbursement"] >= risk_threshold_dict["total_reimbursement"]).astype(int)

final_test_dataset["high_claim_per_patient_flag"] = (final_test_dataset["claim_per_patient"] >= risk_threshold_dict["claim_per_patient"]).astype(int)

final_test_dataset["high_inpatient_claim_ratio_flag"] = (final_test_dataset["inpatient_claim_ratio"] >= risk_threshold_dict["inpatient_claim_ratio"]).astype(int)

final_test_dataset["high_cost_claim_percentage_flag"] = (final_test_dataset["high_cost_claim_percentage"] >= risk_threshold_dict["high_cost_claim_percentage"]).astype(int)

final_test_dataset["high_max_claim_duration_flag"] = (final_test_dataset["max_claim_duration_days"] >= risk_threshold_dict["max_claim_duration_days"]).astype(int)

## Risk Score & Levels

risk_flag_columns = ["high_total_claim_flag", "high_total_reimbursement_flag", "high_claim_per_patient_flag", "high_inpatient_claim_ratio_flag", "high_cost_claim_percentage_flag", "high_max_claim_duration_flag"]

final_test_dataset["risk_score"] = final_test_dataset[risk_flag_columns].sum(axis=1)

def risk_level (risk_score):
    if risk_score >=4: 
        return "High"
    elif risk_score >=2:
        return "Medium"
    else:
        return "Low"

final_test_dataset["risk_level"] = final_test_dataset["risk_score"].apply(risk_level)

print("\nRisk Score Dist")
print(final_test_dataset["risk_score"].value_counts().sort_index())

print("\nRisk Level Dist")
print(final_test_dataset["risk_level"].value_counts())

## Risk level into Test Prediction

test_prediction = test_prediction.merge(final_test_dataset[["Provider", "risk_score", "risk_level"]], on= "Provider", how= "left", validate = "one_to_one")

print("\nTest Risk Score Shape")
print(test_prediction.shape)

print("\nPrediction by Risk Level")
print(pd.crosstab(test_prediction["risk_level"], test_prediction["predict_fraud_status"]))

## Save Data
##=========================================================================================================================

test_prediction_file = OUTPUT_FOLDER / "test_provider_predictions.csv"
test_prediction.to_csv(test_prediction_file, index=False)

print(test_prediction_file)

