from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FOLDER = PROJECT_ROOT / "output" / "models"

MODELS_FOLDER =  OUTPUT_FOLDER / "models"

x_train_file = MODELS_FOLDER / "x_train.csv"
x_test_file = MODELS_FOLDER / "x_test.csv"
y_train_file = MODELS_FOLDER / "y_train.csv"
y_test_file = MODELS_FOLDER / "y_test.csv"

x_train = pd.read_csv(x_train_file)
x_test = pd.read_csv(x_test_file)

y_train = pd.read_csv(y_train_file)["PotentialFraudFlag"]
y_test = pd.read_csv(y_test_file)["PotentialFraudFlag"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
pd.set_option("display.float.format", "{:,.2f}".format)

##==========================================================================================================================
## Model 1 : Logistic Regresssion 
##=========================================================================================================================

logistic_regression_model = Pipeline(steps= [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter= 1000, class_weight= "balanced", random_state=42))])

logistic_regression_model.fit(x_train, y_train)

##==========================================================================================================================
## Checking on Train Data
##==========================================================================================================================

train_pred = logistic_regression_model.predict(x_train)

train_prob = logistic_regression_model.predict_log_proba(x_train)[:,1]

print("\nClassification Report")
print(classification_report(y_train, train_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_train, train_pred))

print("\nROC-AUC")
print(round(roc_auc_score(y_train, train_prob),4))

print("\nPR-AUC")
print(round(average_precision_score(y_train, train_prob),4))

##=========================================================================================================================
## Checking on Test Data
##=========================================================================================================================

test_pred = logistic_regression_model.predict(x_test)

test_prob = logistic_regression_model.predict_proba(x_test)[:,1]

print("\nTest Classification Report")
print(classification_report(y_test, test_pred))

print("\nTest Confusion Matrix")
print(confusion_matrix(y_test, test_pred))

print("\nTest ROC-AUC")
print(round(roc_auc_score(y_test, test_prob),4))

print("\nTest PR-AUC")
print(round(average_precision_score(y_test, test_prob),4))

##=========================================================================================================================
## Save Logistic Regression Model
##========================================================================================================================= 

logistic_regression_model_file = MODELS_FOLDER / "logistic_regression_model.pkl"

joblib.dump(logistic_regression_model, logistic_regression_model_file)

print(logistic_regression_model_file)


##=========================================================================================================================
## Model 2: Decision Tree
##=========================================================================================================================

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=5, class_weight= "balanced", random_state=42)

decision_tree_model.fit(x_train, y_train)

##==========================================================================================================================
## Decision Tree - Checking on Train Data
##==========================================================================================================================

dt_train_pred = decision_tree_model.predict(x_train)

dt_train_prob = decision_tree_model.predict_proba(x_train)[:,1]

print("\nDecision Tree- Classification Report")
print(classification_report(y_train,dt_train_pred))

print("\nDecision Tree - Confusion Matrix")
print(confusion_matrix(y_train,dt_train_pred))

print("\nDecision Tree - ROC-AUC")
print(round(roc_auc_score(y_train, dt_train_prob),4))

print("\nDecision Tree - PR-AUC")
print(round(average_precision_score(y_train, dt_train_prob),4))

##=========================================================================================================================
## Decision Tree - Checking on Test Data
##=========================================================================================================================

dt_test_pred = decision_tree_model.predict(x_test)

dt_test_prob = decision_tree_model.predict_proba(x_test)[:,1]

print("\nDecision Tree - Test Classification Report")
print(classification_report(y_test,dt_test_pred))

print("\nDecision Tree - Test Confusion Matrix")
print(confusion_matrix(y_test,dt_test_pred))

print("\nDecision Tree - Test ROC-AUC")
print(round(roc_auc_score(y_test,dt_test_prob),4))

print("\nDecision Tree- Test PR-AUC")
print(round(average_precision_score(y_test, dt_test_prob),4))

##=========================================================================================================================
## Save Decision Tree Model
##=========================================================================================================================

decision_tree_model_file = MODELS_FOLDER / "decision_tree_model.pkl"

joblib.dump(decision_tree_model, decision_tree_model_file)

print(decision_tree_model_file)


##=========================================================================================================================
## Model 3 - Random Forest
##=========================================================================================================================

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=200, max_depth= 8, class_weight= "balanced", random_state=42)

random_forest_model.fit(x_train, y_train)

##========================================================================================================================
## Random Forest - Checking on Train Data
##========================================================================================================================

rf_train_pred = random_forest_model.predict(x_train)

rf_train_prob = random_forest_model.predict_proba(x_train)[:,1]

print("\nRandom Forest - Classification Report")
print(classification_report(y_train, rf_train_pred))

print("\nRandom Forest - Confusion Matrix")
print(confusion_matrix(y_train,rf_train_pred))

print("\nRandom Forest - ROC-AUC")
print(round(roc_auc_score(y_train,rf_train_prob),4))

print("\nRandom Forest - PR-AUC")
print(round(average_precision_score(y_train, rf_train_prob),4))

##========================================================================================================================
## Random Forest - Checking on Test Data
##========================================================================================================================

rf_test_pred = random_forest_model.predict(x_test)

rf_test_prob = random_forest_model.predict_proba(x_test)[:,1]

print("\nRandom Forest - Test Classification Report")
print(classification_report(y_test, rf_test_pred))

print("\nRandom Forest - Test Confusion Matrix")
print(confusion_matrix(y_test, rf_test_pred))

print("\nRandom Forest - Test ROC-AUC")
print(round(roc_auc_score(y_test, rf_test_prob),4))

print("\nRandom Forest - Test PR-AUC")
print(round(average_precision_score(y_test, rf_test_prob),4))

##=========================================================================================================================
## Save Random Forest Model
##=========================================================================================================================

random_forest_model_file = MODELS_FOLDER / "random_forest_model.pkl"

joblib.dump(random_forest_model, random_forest_model_file)

print(random_forest_model_file)


##==========================================================================================================================
## Model 4 - Gradient Boosting
##==========================================================================================================================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

gradient_boosting_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

gradient_boosting_sample_weights = compute_sample_weight(class_weight="balanced", y =y_train)

gradient_boosting_model.fit(x_train, y_train, sample_weight= gradient_boosting_sample_weights)

##==========================================================================================================================
## Gradient Boosting - Checking on Train Data
##==========================================================================================================================

gb_train_pred = gradient_boosting_model.predict(x_train)

gb_train_prob= gradient_boosting_model.predict_proba(x_train)[:,1]

print("\nGradient Boosting - Classification Report")
print(classification_report(y_train, gb_train_pred))

print("\nGradient Boosting - Confusion Matrix")
print(confusion_matrix(y_train, gb_train_pred))

print("\nGradient Boosting - ROC-AUC")
print(round(roc_auc_score(y_train, gb_train_prob),4))

print("\nGradient Boosting - PR-AUC")
print(round(average_precision_score(y_train, gb_train_prob),4))

##=========================================================================================================================
## Gradient Boosting - Checking on Test Data
##========================================================================================================================

gb_test_pred = gradient_boosting_model.predict(x_test)

gb_test_prob = gradient_boosting_model.predict_proba(x_test)[:,1]

print("\nGradient Boosting - Test Classification Report")
print(classification_report(y_test,gb_test_pred))

print("\nGradient Boosting - Test Confusion Report")
print(confusion_matrix(y_test, gb_test_pred))

print("\nGradient Boosting - Test ROC-AUC")
print(round(roc_auc_score(y_test, gb_test_prob),4))

print("\nGradient Boosting - Test PR-AUC")
print(round(average_precision_score(y_test, gb_test_prob),4))

##=========================================================================================================================
## Save Gradient Boosting Model
##=========================================================================================================================

gradient_boosting_model_file = MODELS_FOLDER / "gradient_boosting_model.pkl"

joblib.dump(gradient_boosting_model, gradient_boosting_model_file)

print(gradient_boosting_model_file)

##=========================================================================================================================
## Model Evaluation Summary
##=========================================================================================================================

model_performance_summary = pd.DataFrame([{"model_name": "Logistic Regression",
            "accuracy": 0.90,
            "fraud_precision": 0.47,
            "fraud_recall": 0.89,
            "fraud_f1_score": 0.62,
            "roc_auc": 0.9669,
            "pr_auc": 0.7892,
            "true_negatives": 880,
            "false_positives": 101,
            "false_negatives": 11,
            "true_positives": 90},
            { "model_name": "Decision Tree",
            "accuracy": 0.88,
            "fraud_precision": 0.43,
            "fraud_recall": 0.87,
            "fraud_f1_score": 0.58,
            "roc_auc": 0.8752,
            "pr_auc": 0.6626,
            "true_negatives": 864,
            "false_positives": 117,
            "false_negatives": 13,
            "true_positives": 88},
            {"model_name": "Random Forest",
            "accuracy": 0.91,
            "fraud_precision": 0.52,
            "fraud_recall": 0.88,
            "fraud_f1_score": 0.65,
            "roc_auc": 0.9658,
            "pr_auc": 0.7635,
            "true_negatives": 898,
            "false_positives": 83,
            "false_negatives": 12,
            "true_positives": 89},
            {"model_name": "Gradient Boosting",
             "accuracy": 0.91,
             "fraud_precision": 0.51,
             "fraud_recall": 0.90,
             "fraud_f1_score": 0.65,
             "roc_auc": 0.9674,
             "pr_auc": 0.7825,
             "true_negatives": 895,
             "false_positives": 86,
             "false_negatives": 10,
             "true_positives": 91}])

print("\nModel Summary")
print(model_performance_summary.to_string(index=False))

##=========================================================================================================================
## Ranking Models
##=========================================================================================================================

ranked_models = (model_performance_summary.sort_values(by=["fraud_recall", "fraud_f1_score", "pr_auc", "roc_auc"], ascending= False))

print("\nRanking Models")
print(ranked_models.to_string(index=False))

##==========================================================================================================================
## Best Model
##==========================================================================================================================

selected_model_name = ranked_models.iloc[0]["model_name"]

print("\nBest Model")
print(selected_model_name)

##=========================================================================================================================
## Model Save
##========================================================================================================================

model_performance_summary_file = OUTPUT_FOLDER / "model_performance_summary.csv"

selected_model_summary_file = OUTPUT_FOLDER / "selected_model_summary.txt"

model_performance_summary.to_csv(model_performance_summary_file, index=False)
print(model_performance_summary_file)

selected_model_summary = f"""
Select Model: {selected_model_name}

Reason:
Gradient Boosting was selected because it achieved the strongest overall fraud-screening performance on validation data.

Validation Performance:
- Fraud precision: 0.51
- Fraud recall: 0.90
- Fraud F1-score: 0.65
- ROC-AUC: 0.9674
- PR-AUC: 0.7825

Interpretation:
The model caught 91 out of 101 fraud-labeled providers in the validation dataset and missed only 10 fraud-labeled providers. It also had a strong ROC-AUC and PR-AUC, making it a strong model for provider fraud-risk screening.

Important Note:
This model should be used as a risk-screening tool, not as proof of fraud.
"""

with open(selected_model_summary_file, "w") as file: file.write(selected_model_summary)

print(selected_model_summary_file)