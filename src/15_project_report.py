from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "output"
CHARTS_FOLDER = OUTPUT_FOLDER / "charts"
REPORTS_FOLDER = OUTPUT_FOLDER / "reports"

REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)

prediction_file = OUTPUT_FOLDER / "predictions" / "test_provider_predictions.csv"
test_predictions = pd.read_csv (prediction_file, low_memory=False)

model_performance_file = OUTPUT_FOLDER / "models" / "model_performance_summary.csv "
model_performance = pd.read_csv(model_performance_file, low_memory=False)

selected_model_summary_file = OUTPUT_FOLDER / "models" / "selected_model_summary.txt"
with open(
    selected_model_summary_file,
    "r"
) as file:
    selected_model_summary = file.read()

final_report_file = REPORTS_FOLDER / "final_healthcare_fraud_detection_report_v2.pdf"

print("\nShape")
print(test_predictions.shape)
print(model_performance.shape)

print("\nmodel")
print(selected_model_summary)

print(final_report_file)

##=======================================
## 
##=======================================

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

##=======================================
## Report Summary Numbers
##======================================

status_column = "predict_fraud_status"

if status_column not in test_predictions.columns:
    status_column = "predict_fruad_status"

total_test_providers = test_predictions.shape[0]

prediction_counts = test_predictions[status_column].value_counts()
risk_level_counts = test_predictions["risk_level"].value_counts()

potential_fraud_count = prediction_counts.get("Potential Fraud", 0)

not_potential_fruad_count = prediction_counts.get("Not Potential Fraud", 0)

high_risk_count = risk_level_counts.get("High", 0)

mediun_risk_count = risk_level_counts.get("Medium", 0)

low_risk_count = risk_level_counts.get("Low", 0)

print("\nTotal test provider")
print(total_test_providers)

print("\nPrediction counts")
print(prediction_counts)

print("\nRisk level counts")
print(risk_level_counts)

##==========================
## Basic PDF Report
##===========================

pdf_doc = SimpleDocTemplate(str(final_report_file), pagesize= letter, rightMargin=40, leftMargin=40,topMargin=40,bottomMargin=40)

styles= getSampleStyleSheet()

title = ParagraphStyle(name="TitleStyle", parent=styles["Title"], fontSize=18,spaceAfter=18)

heading = ParagraphStyle(name= "HeadingStyle", parent=styles["Heading2"],fontSize=13,spaceBefore=12,spaceAfter=8)

body= ParagraphStyle(name="BodyStyle", parent=styles["BodyText"],fontSize=10,leading=14)

story= []

story.append(Paragraph("Healthcare Provider Fraud Detection Project Report", title))

story.append(Paragraph("This report summarizes a healthcare claims fraud detection project using provider-level claims, beneficiary, billing, demographic, chronic-condition, and risk-based features.", body))

story.append(Spacer(1,12))

story.append(Paragraph("1. Project Objective", heading))

story.append(Paragraph("The objective of this project is to identify healthcare providers with elevated potential fraud risk using historical claims and beneficiary-level data. The model is designed as a fraud-risk screening tool, not as proof of fraud.", body))

story.append(Paragraph("2. Test Prediction Summary", heading))

summary_table = [["Metric", "Value"], ["Total test providers", total_test_providers], ["Predicted potential fraud providers", potential_fraud_count], ["Predicted not potential fraud providers", not_potential_fruad_count],
                 ["High risk providers", high_risk_count], ["Medium risk providers", mediun_risk_count], ["Low risk providers", low_risk_count]]

summary_table= Table(summary_table, hAlign="LEFT")

summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1),9),
    ("BOTTOMPADDING", (0,0), (-1,-1),6),
    ("TOPPADDING",(0,0), (-1,-1),6 )]))

story.append(summary_table)


print("\nBasic PDF report")
print(final_report_file)



##=================================
## Methodology Sections
##=================================

story.append(Paragraph("3. Data Used",heading))

story.append(Paragraph("The project used provider-level, claims-level, and beneficiary-level healthcare datasets. The provider dataset contained billing and utilization summaries. The claims dataset contained individual inpatient and outpatient claims. The beneficiary dataset contained patient demographic and chronic-condition information.",body))


story.append(Paragraph("4. Data Cleaning",heading))

story.append(Paragraph("The claims data was cleaned by converting claim start and end dates into datetime format, checking invalid claim durations, handling missing deductible amounts, and validating negative reimbursement or deductible values. Beneficiary data was cleaned by converting date of birth and date of death fields into datetime format, encoding chronic-condition indicators, and encoding renal disease indicators. Provider-level data was checked for duplicates, missing values, and column-name consistency.",body))


story.append(Paragraph("5. Feature Engineering",heading))

story.append(Paragraph("Provider-level features were created from claims and beneficiary data. These included demographic features such as average patient age, gender distribution, and race distribution; chronic-condition features such as average chronic disease burden and disease-specific rates; and fraud-risk features such as claim duration, high-cost claim percentage, reimbursement intensity, and inpatient claim ratio.",body))


story.append(Paragraph( "6. Rule-Based Risk Levels",heading))

story.append(Paragraph("A rule-based risk score was created using selected high-risk indicators. Providers received one point for exceeding training-data thresholds on features such as total claims, total reimbursement, claim-per-patient ratio, reimbursement per patient, inpatient claim ratio, high-cost claim percentage, and maximum claim duration. Providers were categorized as Low, Medium, or High risk based on their total risk score.",body))



##=======================
## Model Performance Section
##==========================

story.append(Paragraph("7. Model Training and Comparison", heading))

story.append(Paragraph("Four machine learning models were trained and compared: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The models were evaluated using fraud precision, fraud recall, fraud F1-score, ROC-AUC, PR-AUC. Since fraud detection is an imbalanced classification problem, recall, F1-score, and PR-AUC were especially important.",body))

model_table = [["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]

for index, row in model_performance.iterrows():
    model_table.append([row["model_name"],
                            round(row["fraud_precision"],2),
                            round(row["fraud_recall"],2),
                            round(row["fraud_f1_score"],2),
                            round(row["roc_auc"],4),
                            round(row["pr_auc"],4 )])

model_table = Table(model_table, hAlign="LEFT")

model_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                                 ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                                 ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                                 ("FONTSIZE", (0,0), (-1,-1), 8),
                                 ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                                 ("TOPPADDING", (0,0), (-1,-1),5)]))

story.append(model_table)

##=================================
## Selected Model
##================================

story.append(Paragraph("8. Selected Model", heading))

selected_model_summary_pdf = selected_model_summary.replace("\n", "<br/>")

story.append(Paragraph(selected_model_summary_pdf, body))

##============================
## Test Prediction Interpretation
##===============================

story.append(Paragraph("9. Test Prediction Results", heading))

story.append(Paragraph(f"The selected Gradient Boosting model was applied to {total_test_providers} actual test providers. The model predicted {potential_fraud_count} providers as potential fruad and {not_potential_fruad_count} providers as not potential fraud.", body))

story.append(Spacer(1,8))

prediction_risk_table = [["Risk Level","Not Potential Fraud","Potential Fraud"]]

prediction_by_risk = pd.crosstab(test_predictions["risk_level"], test_predictions[status_column])

for risk_level_name in ["High", "Medium", "Low"]:
    if risk_level_name in prediction_by_risk.index:
        prediction_risk_table.append([risk_level_name, prediction_by_risk.loc[risk_level_name].get("Not Potential Fraud", 0), prediction_by_risk.loc[risk_level_name].get("Potential Fraud", 0)])

prediction_risk_table = Table(prediction_risk_table, hAlign="LEFT")

prediction_risk_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5)]))

story.append(prediction_risk_table)

##=================================
## Charts Section
##===================================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

story.append(PageBreak())

story.append(Paragraph("10. Project Visualizations", heading))

story.append(Paragraph("The following charts summarize the final test predictions, risk levels, model behavior, and model comparison results.",body))

story.append(Spacer(1,10))

chart_files = [("Predicted Fraud Status Distribution", CHARTS_FOLDER / "test_fraud_status_dist_test.png"),
    ("Risk Level Distribution", CHARTS_FOLDER / "test_risk_level_dist_test.png"),
    ("Fraud Prediction By Risk Level",CHARTS_FOLDER / "test_pred_by_risk.png"),
    ("Fraud Probability Distribution",CHARTS_FOLDER / "test_fraud_prob_distribution.png"),
    ("Top 20 Providers By Fraud Probability", CHARTS_FOLDER / "top_20_test_provider_fraud.png"),
    ("Top 20 Model Feature Importance", CHARTS_FOLDER / "top_20_model_importance.png"),
    ("Model Performance Comparison",CHARTS_FOLDER / "model_comparison.png")]

for chart_title, chart_file in chart_files:
    if chart_file.exists():

        chart_image= Image(str(chart_file))

        max_chart_width = 360
        max_chart_height = 180

        width_scale = max_chart_width / chart_image.imageWidth
        height_scale = max_chart_height / chart_image.imageHeight

        scale = min( width_scale, height_scale)

        chart_image.drawWidth = chart_image.imageWidth * scale
        chart_image.drawHeight = chart_image.imageHeight * scale

        chart_block = KeepTogether([Paragraph(chart_title, heading), Spacer(1,6), chart_image, Spacer(1,12)])
        story.append(chart_block)
    else:
        story.append(Paragraph(f"Chart not found: {chart_file.name}", body))


##===========================
## Limitations and Conclusion
##============================


story.append(Paragraph("11. Limitations", heading))

story.append(Paragraph("This project identifies potential fraud risk, not confirmed fraud. The dataset does not contain investigation outcomes or legal findings. The model is based on available claims, reimbursement, provider, demographic, and chronic-condition features. Diagnosis and procedure-code level fraud patterns could not be analyzed because those fields were not available in the combined claims dataset used for this project. Model predictions should therefore be used as a screening tool for prioirtizing review, not as final evidence of fraud.", body))

story.append(Paragraph("12. Conclusion", heading))

story.append(Paragraph(f"The final Gradient Boosting model was seleted because it provided strong validation performance, especially fraud recall and fraud F1-score. On the actual test provider dataset, the model predicted {potential_fraud_count} out of {total_test_providers} providers as potential fraud. The rule-based risk-level analysis also helped explain the model output by showing how predicted fraud providers were distributed across Low, Medium, and High risk categories.", body))

story.append(Spacer(1,12))

story.append(Paragraph("Overall, the project demonstrates an end-to-end healthcare fraud detection workflow covering data cleaning, feature engineering, exploratory analysis, risk scoring, machine learning model training, test-data prediction, visualization, and reporting.", body))


print(len(story))

print("\nChart file")
for chart_title, chart_file in chart_files:
    print(chart_file.name, chart_file.exists())

pdf_doc.build(story)

print(final_report_file)