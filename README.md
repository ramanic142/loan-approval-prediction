# Loan Approval Prediction
# Logistic Regression vs Neural Network | R

![R](https://img.shields.io/badge/Language-R-276DC3?style=flat&logo=r)
![ML](https://img.shields.io/badge/Type-Classification-brightgreen)
![Dataset](https://img.shields.io/badge/Records-4269-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

#  Overview

Built a binary classification system to predict loan approval outcomes based on applicant financial and demographic data. Compared **Logistic Regression** (interpretable baseline) against a **Neural Network** (MLP) using ROC-AUC and confusion matrix evaluation.

#  Dataset

| Detail | Value |
|---|---|
| Source | Kaggle — Loan Approval Prediction Dataset |
| Records | 4,269 applications |
| Features | 13 (+ 2 engineered) |
| Approval Rate | 61.12% |
| Train / Test Split | 75% / 25% |

**Engineered Features:**
- `total_assets` = sum of all 4 asset values
- `loan_to_income` = loan amount ÷ annual income

#  Exploratory Data Analysis

<table>
  <tr>
    <td><img src="images/slide-07.jpg" width="400"/><br><sub>CIBIL Score by Loan Status</sub></td>
    <td><img src="images/slide-06.jpg" width="400"/><br><sub>Income vs Loan Amount</sub></td>
  </tr>
  <tr>
    <td><img src="images/slide-08.jpg" width="400"/><br><sub>Approval Rate by Education</sub></td>
    <td><img src="images/slide-09.jpg" width="400"/><br><sub>Approval Rate by Employment Type</sub></td>
  </tr>
</table>

**Correlation Matrix**

<img src="images/slide-10.jpg" width="500"/>


#  Models Built

# 1. Logistic Regression
- Binary classification with binomial family
- VIF test for multicollinearity
- Hosmer-Lemeshow goodness-of-fit test
- Cook's Distance for outlier detection
- Odds ratio interpretation for business explainability

# 2. Neural Network (MLP)
- 3 hidden nodes
- Trained on 6 key numerical features
- Compared against logistic regression via ROC curve

# Results
# Logistic Regression
<table>
  <tr>
    <td><img src="images/slide-16.jpg" width="400"/><br><sub>Confusion Matrix</sub></td>
    <td><img src="images/slide-17.jpg" width="400"/><br><sub>ROC Curve</sub></td>
  </tr>
</table>

# Neural Network
<table>
  <tr>
    <td><img src="images/slide-20.jpg" width="400"/><br><sub>Confusion Matrix</sub></td>
    <td><img src="images/slide-21.jpg" width="400"/><br><sub>ROC Curve</sub></td>
  </tr>
</table>

# ROC Comparison
<img src="images/slide-22.jpg" width="500"/>

#  Model Performance

<img src="images/slide-25.jpg" width="500"/>

| Model | AUC Score | Preferred? |
|---|---|---|
| Logistic Regression | 0.548 | Yes |
| Neural Network | 0.549 | No |

> Logistic Regression is preferred — nearly identical performance with far better interpretability for banking contexts.

#  Key Findings

- **CIBIL score** is the strongest predictor of loan approval
- **Self-employed applicants** have lower approval odds than salaried
- **High loan-to-income ratio** significantly reduces approval probability
- **Graduate applicants** show higher approval rates
- Neural Network adds complexity without meaningful accuracy gain

#  How to Run

```r
# Install packages (first time only)
install.packages(c("readr", "dplyr", "ggplot2", "reshape2",
                   "caret", "car", "pROC", "ResourceSelection",
                   "neuralnet", "scales"))

# Run the script
source("loan_approval.R")
```
#  Repository Structure
```
loan-approval-prediction/
├── loan_approval.R               # R script
├── loan_approval_dataset.csv     # Dataset
├── images/                       # Result visualizations
└── README.md
```
#  Tech Stack
`R` `ggplot2` `caret` `pROC` `neuralnet` `car` `ResourceSelection`
