# Term-Deposit-Subscription-Prediction
End-to-end machine learning pipeline for predicting bank term deposit subscriptions using statistical analysis, business insights, and classification models.

A machine learning project that predicts whether a customer will subscribe to a bank term deposit using customer demographics, financial attributes, and previous marketing campaign data.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Objectives](#-objectives)
- [Tech Stack](#️-tech-stack)
- [Results](#-results)
- [Installation & Usage](#️-installation--usage)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📌 Project Overview

Marketing campaigns are expensive, and banks often contact thousands of customers with low conversion rates. This project builds an end-to-end machine learning pipeline using the UCI Bank Marketing dataset to predict which customers are likely to subscribe to a term deposit — helping banks target the right customers, cut wasted outreach, and improve campaign ROI.

The pipeline covers exploratory data analysis, statistical diagnostics, feature engineering, handling class imbalance, training and tuning multiple classification models, and translating model output into a business-ready customer scoring system.

## 🏦 Business Problem

Banks invest significant resources in telemarketing campaigns to promote term deposit products. However, contacting every customer is costly and often results in low response rates.

The objective of this project is to build a predictive model that identifies customers who are more likely to subscribe to a term deposit. Such predictions can help banks:

- Improve campaign targeting
- Reduce unnecessary marketing costs
- Increase conversion rates
- Support data-driven marketing strategies


## 📊 Dataset

**Source:** [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

- **Records:** 41,188 customer entries
- **Features:** 21 input variables (demographic, financial, campaign-related, and macroeconomic indicators)
- **Target:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- **Class balance:** ~11% positive class (subscribed), ~89% negative — a real-world imbalanced classification problem
- **File used:** `bank-additional-full.csv`
- **Origin:** Direct marketing (phone call) campaigns of a Portuguese banking institution

## 🔍 Methodology

**1. Exploratory Data Analysis**
Examined distributions of customer demographics (age, job, marital status, education), financial attributes (loan/housing/default status), campaign behavior (contact method, month, call duration, number of contacts), and macroeconomic indicators (employment variation rate, consumer price/confidence index, euribor rate). Checked feature correlations and target class balance.

**2. Preprocessing & Feature Engineering**
- One-hot encoded categorical variables, scaled numerical features with `StandardScaler`
- Checked multicollinearity using Variance Inflation Factor (VIF) and removed redundant features
- Verified linearity of the logit for continuous predictors using the Box-Tidwell test

**3. Handling Class Imbalance**
Since only ~11% of customers subscribed, standard classifiers were biased toward the majority class. Addressed this using `EasyEnsembleClassifier` and `BalancedRandomForestClassifier` from `imbalanced-learn`, which resample the majority class during ensemble training.

**4. Model Building & Hyperparameter Tuning**
Trained and tuned multiple classifiers — Logistic Regression, SVM, Decision Tree, Random Forest, Balanced Random Forest, AdaBoost-style ensemble, Gradient Boosting, and XGBoost — using **Optuna** for Bayesian hyperparameter search (and `GridSearchCV` for the decision tree baseline).

**5. Model Evaluation & Ensembling**
Compared all models on accuracy, precision, recall, F1-score, and ROC-AUC, prioritizing recall/F1 over raw accuracy given the class imbalance. Combined the best-tuned models into hard and soft Voting Classifiers.

**6. Business Scoring**
Used the final model's predicted probabilities to bucket customers into subscription-likelihood tiers (e.g., >90%, 80–90%, 70–80%, etc.), enabling banks to prioritize outreach to the highest-probability customers.

## 🎯 Objectives

- Analyze customer, campaign, and economic factors influencing term deposit subscriptions
- Build and compare multiple classification models to predict subscription likelihood
- Address class imbalance to ensure reliable predictions on the minority (subscribing) class
- Identify the most important predictors driving customer subscription decisions
- Translate model predictions into a practical customer scoring system for targeted marketing

## 🛠️ Tech Stack

**Language:** Python

**Data Handling & Analysis:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn, Plotly

**Statistical Analysis:** Statsmodels (VIF, Logit, Box-Tidwell test)

**Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (EasyEnsemble, Balanced Random Forest)

**Hyperparameter Tuning:** Optuna

**Environment:** Jupyter Notebook

## 📈 Results

Models were evaluated on accuracy, precision, recall, F1-score, and ROC-AUC, with a focus on **recall and F1** given the ~11% positive class imbalance.

| Model                        | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression           | [X]      | [X]       | [X]    | [X]      | [X]     |
| SVM (Optuna-tuned)             | [X]      | [X]       | [X]    | [X]      | [X]     |
| Decision Tree (Optuna-tuned)   | [X]      | [X]       | [X]    | [X]      | [X]     |
| Balanced Random Forest (Optuna)| [X]      | [X]       | [X]    | [X]      | [X]     |
| AdaBoost-style Ensemble        | [X]      | [X]       | [X]    | [X]      | [X]     |
| Gradient Boosting (Optuna)     | [X]      | [X]       | [X]    | [X]      | [X]     |
| **XGBoost (Optuna-tuned)**     | **[X]**  | **[X]**   | **[X]**| **[X]**  | **[X]** |
| Voting Ensemble (Soft)         | 0.864      | 0.457       | 0.92    | 0.612      | 0.891    |

**Best model:** [Name the model with the best recall/F1/ROC-AUC tradeoff] was selected as the final model based on [your reasoning — e.g., "highest ROC-AUC while maintaining strong recall on the minority class"].


## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/RohitSen0701/Term-Deposit-Subscription-Prediction.git
cd Term-Deposit-Subscription-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**
Download `bank-additional-full.csv` from the [UCI Bank Marketing dataset page](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it in the project folder.

**4. Run the notebook**
```bash
jupyter notebook Term-Deposit-Prediction.ipynb
```
## 🚀 Future Improvements

- Deploy the model using Streamlit or Flask.
- Integrate SHAP for model explainability.
- Build an interactive dashboard for customer scoring.
- Automate model retraining using new campaign data.

## 👨‍💻 Author

**Rohit Sen**

M.Sc. Statistics, University of Delhi

LinkedIn:
https://www.linkedin.com/in/rohit-sen-50188b320/

GitHub:
https://github.com/RohitSen0701
