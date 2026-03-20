import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. DATA LOADING & INITIAL AUDIT ---
url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
df = pd.read_csv(url, sep='\t')

print(f"Dataset Shape: {df.shape}")
print(f"Duplicates: {df.duplicated().sum()}")

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='BMI', color='skyblue').set_title('BMI Distribution (Outlier Check)')
plt.subplot(1, 2, 2)
sns.boxplot(data=df, y='BP', color='salmon').set_title('BP Distribution (Variance Check)')
plt.show()

# Correlation Matrix (Identifying Multicollinearity)
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='RdBu_r', mask=mask)
plt.title('Feature Correlation Heatmap')
plt.show()

# --- 3. DATA PREPROCESSING (The "Anti-Leakage" Way) ---
X = df.drop("Y", axis=1)
y = df["Y"]

# Split first, scale second
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to DataFrame for easier VIF calculation
X_train_df = pd.DataFrame(X_train, columns=X.columns)

# --- 4. MANUAL IMPLEMENTATION (Normal Equation) ---
# Formula: theta = (X.T @ X)^-1 @ X.T @ y
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train] # Add intercept
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

print(f"\nManual Intercept: {theta_best[0]:.4f}")
print(f"Manual Coefficients (BMI): {theta_best[3]:.4f}")

# --- 5. MODEL DIAGNOSTICS: VIF ---
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
print("\nVariance Inflation Factors (Multicollinearity Check):")
print(vif_data.sort_values(by="VIF", ascending=False))

# --- 6. MODEL TRAINING & REGULARIZATION ---
models = {
    "OLS": LinearRegression(),
    "Lasso": LassoCV(cv=5),
    "Ridge": RidgeCV(cv=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    # Adjusted R2
    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    results[name] = {"R2": r2, "Adj_R2": adj_r2, "RMSE": rmse, "MAE": mae}

# Display Results
results_df = pd.DataFrame(results).T
print("\nFinal Model Performance Comparison:")
print(results_df)

# --- 7. RESIDUAL ANALYSIS (Using Lasso for the Best Fit) ---
best_model = models["Lasso"]
final_preds = best_model.predict(X_test)
residuals = y_test - final_preds

plt.figure(figsize=(14, 5))

# Residuals vs Fitted
plt.subplot(1, 2, 1)
sns.scatterplot(x=final_preds, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted (Homoscedasticity)')

# Q-Q Plot
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot (Residual Normality)')
plt.show()

# --- 8. CLINICAL INSIGHTS: FEATURE IMPORTANCE ---
importance = pd.Series(np.abs(best_model.coef_), index=X.columns).sort_values()
plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='teal')
plt.title('Clinical Feature Importance (Absolute Lasso Coefficients)')
plt.xlabel('Impact on Disease Progression')
plt.show()

