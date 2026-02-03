# Roll No: 24BAD047
# Multilinear Regression - Student Performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\4_sem\ML\EX_NO_3\StudentsPerformance.csv")

# 2. Create target variable (average score)
df["final_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3

# 3. Encode categorical features
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["parental level of education"] = le.fit_transform(df["parental level of education"])
df["test preparation course"] = le.fit_transform(df["test preparation course"])
df["lunch"] = le.fit_transform(df["lunch"])
df["race/ethnicity"] = le.fit_transform(df["race/ethnicity"])

# 4. Select features
X = df[[
    "gender",
    "parental level of education",
    "test preparation course",
    "lunch",
    "race/ethnicity"
]]
y = df["final_score"]

# 5. Handle missing values
X.fillna(X.mean(), inplace=True)

# 6. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Multilinear Regression Results")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 11. Coefficients
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nCoefficients:\n", coeff_df)

# 12. Ridge and Lasso
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print("\nRidge R2:", ridge.score(X_test, y_test))
print("Lasso R2:", lasso.score(X_test, y_test))

# ----------- Visualizations -----------

# Predicted vs Actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Scores")
plt.show()

# Coefficient plot
sns.barplot(x="Coefficient", y="Feature", data=coeff_df)
plt.title("Feature Coefficients")
plt.show()

# Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
