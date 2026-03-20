# Diabetic_Linear_Regression_Model
# 🩺 Predicting Diabetes Progression: An Advanced Regression Study

This project applies supervised machine learning to predict disease progression in diabetes patients using 10 baseline variables. The study focuses on model diagnostics, handling multicollinearity, and comparing OLS with Regularized Regression (Lasso/Ridge).

## 🚀 Project Overview
The goal was to build a predictive model for disease progression while ensuring the model adheres to the fundamental assumptions of linear regression. 

### Key Technical Features:
* **Mathematical Implementation**: Linear Regression from scratch using the Normal Equation: $\theta = (X^T X)^{-1} X^T y$.
* **Diagnostic Suite**: VIF (Variance Inflation Factor), Residual Analysis (Normality/Homoscedasticity), and Correlation Mapping.
* **Regularization**: Lasso (L1) and Ridge (L2) with 5-fold Cross-Validation to handle high feature redundancy.
* **Preprocessing**: Implementation of a "leakage-free" scaling pipeline.

## 📊 Key Findings
* **Multicollinearity**: Detected severe redundancy in blood serum markers (S1 VIF: 55.25).
* **Model Performance**: Lasso Regression outperformed OLS, achieving an **Adjusted R² of 0.40**, proving that feature selection via L1 penalty improves generalization.
* **Top Predictors**: **BMI** and **S5 (Blood Serum Marker)** emerged as the most significant clinical indicators of disease progression.



## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Libraries**: Pandas, NumPy, Scikit-Learn, Statsmodels, Seaborn, Matplotlib

## 🧪 Model Diagnostics
The model was validated using:
1.  **Q-Q Plots**: To ensure residuals follow a normal distribution.
2.  **Residual vs. Fitted Plots**: To verify constant variance (Homoscedasticity).
3.  **VIF Analysis**: To quantify and address feature inflation.

## 🏁 Conclusion
By addressing the multicollinearity in medical serum markers, we improved model accuracy and interpretability. For clinical practitioners, this model emphasizes that monitoring BMI and specific blood markers (S5) provides the highest predictive value for 1-year disease progression.
