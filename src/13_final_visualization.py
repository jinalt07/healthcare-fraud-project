from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FOLDER = PROJECT_ROOT / "output"

CHARTS_FOLDER = OUTPUT_FOLDER / "charts"

prediction_file = OUTPUT_FOLDER / "predictions" / "test_provider_predictions.csv"

test_prediction = pd.read_csv(prediction_file, low_memory= False)

print("\nTest Prediction Shape")
print(test_prediction.shape)

print("\nTest Prediction Columns")
print(test_prediction.columns.tolist())

print("\nPrediction Status Counts")
print(test_prediction["predict_fraud_status"].value_counts())

print("\nRisk Level Counts")
print(test_prediction["risk_level"].value_counts())

##=========================================================================================================================
## Chart 1: Predicted Fraud Status Distribution
##=========================================================================================================================

fruad_status_counts = test_prediction["predict_fraud_status"].value_counts()

plt.figure(figsize=(8,5))

fruad_status_counts.plot(kind="bar")

plt.title("Predicted Fraud Status Distribution")
plt.xlabel("Predicted Fraud Status")
plt.ylabel("Number of Providers")
plt.xticks(rotation=0)
plt.tight_layout()

fruad_status_counts_chart_file = CHARTS_FOLDER / "test_fraud_status_dist_test.png"
plt.savefig(fruad_status_counts_chart_file)
plt.show()
plt.close()
print(fruad_status_counts_chart_file)

##=========================================================================================================================
## Chart 2: Risk Level Distribution
##=========================================================================================================================

risk_level_counts = test_prediction["risk_level"].value_counts()
plt.figure(figsize=(8,5))
risk_level_counts.plot(kind="bar")

plt.title("Test Provider Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Providers")
plt.xticks(rotation=0)
plt.tight_layout()

risk_level_counts_chart_file = CHARTS_FOLDER / "test_risk_level_dist_test.png"

plt.savefig(risk_level_counts_chart_file)
plt.show()
plt.close()
print(risk_level_counts_chart_file)

##=========================================================================================================================
## Chart 3: Fraud Prediction by Risk Level
##=========================================================================================================================

prediction_by_risk = pd.crosstab(test_prediction["risk_level"], test_prediction["predict_fraud_status"])

print("\nPrediction by risk level")
print(prediction_by_risk)

prediction_by_risk.plot(kind="bar", figsize=(8,5))

plt.title("Fraud Prediction By Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Number of Providers")
plt.xticks(rotation=0)
plt.tight_layout()

prediction_by_risk_chart_file = CHARTS_FOLDER / "test_pred_by_risk.png"
plt.savefig(prediction_by_risk_chart_file)
plt.show()
plt.close()
print(prediction_by_risk_chart_file)

##=========================================================================================================================
## Chart 4 : Fraud Probability Distribution
##=========================================================================================================================

plt.figure(figsize=(8,5))

plt.hist(test_prediction["predict_fraud_prob"], bins=30)

plt.title("Test Provider Fraud Probability Distribution")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Number of Providers")
plt.tight_layout()

fruad_probability_chart_file = CHARTS_FOLDER / "test_fraud_prob_distribution.png"

plt.savefig(fruad_probability_chart_file)
plt.show()
plt.close()
print(fruad_probability_chart_file)

##===========================================================================================================================
## Chart 5 : top 20 highest fraud probability
##=========================================================================================================================

top_20_fraud_probability = (test_prediction.sort_values(by= "predict_fraud_prob", ascending = False).head(20))

plt.figure(figsize=(10,6))

plt.barh(top_20_fraud_probability["Provider"], top_20_fraud_probability["predict_fraud_prob"])

plt.title("Top 20 Providers By Predicted Fraud Probability")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Provider")
plt.gca().invert_yaxis()
plt.tight_layout()

top_20_fraud_chart_file = CHARTS_FOLDER / "top_20_test_provider_fraud.png"

plt.savefig(top_20_fraud_chart_file)
plt.show()
plt.close()
print(top_20_fraud_chart_file)

##=========================================================================================================================
## Chart 6 : Model Feature Importance
##=========================================================================================================================

import joblib

model_file = OUTPUT_FOLDER / "models" / "gradient_boosting_model.pkl"

x_train_file = OUTPUT_FOLDER / "models" / "x_train.csv"

selected_model = joblib.load(model_file)

x_train = pd.read_csv(x_train_file, low_memory=False)

feature_importance = pd.DataFrame({"feature": x_train.columns, "importance" : selected_model.feature_importances_})

feature_importance = feature_importance.sort_values(by = "importance", ascending= False)

top_20_features = feature_importance.head(20)

plt.figure(figsize=(10,6))

plt.barh(top_20_features["feature"], top_20_features["importance"])

plt.title("Top 20 Most Important Model Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

feature_importance_chart_file = CHARTS_FOLDER / "top_20_model_importance.png"

plt.savefig(feature_importance_chart_file)
plt.show()
plt.close()
print(feature_importance_chart_file)
print(top_20_features)

##==========================================================================================================================
## Chart 7 : Model Comparision
##==========================================================================================================================

model_performance_file = OUTPUT_FOLDER / "models" / "model_performance_summary.csv"

model_performance = pd.read_csv(model_performance_file, low_memory=False)

print("\nModel Performace Summary")
print(model_performance)

model_comparision = model_performance[["model_name", "fraud_precision", "fraud_recall", "fraud_f1_score", "roc_auc", "pr_auc"]]

model_comparision = model_comparision.set_index("model_name")

model_comparision.plot(kind="bar", figsize=(10,6))

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

model_comparision_chart_file = CHARTS_FOLDER / "model_comparision.png"
plt.savefig(model_comparision_chart_file)
plt.show()
plt.close()
print(model_comparision_chart_file)