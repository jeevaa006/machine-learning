# Roll No: 24BAD047
# Polynomial Regression - Auto MPG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\4_sem\ML\EX_NO_3\auto-mpg.csv")

# 2. Convert horsepower to numeric
df["horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')

# 3. Handle missing values (ONLY numeric columns)
df.fillna(df.select_dtypes(np.number).mean(), inplace=True)

# 4. Select feature and target
X = df[["horsepower"]]
y = df["mpg"]

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

degrees = [2, 3, 4]

for d in degrees:
    print(f"\nPolynomial Degree: {d}")
    
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)
    
    # Plot curve
    X_range = np.linspace(X_scaled.min(), X_scaled.max(), 100)
    X_range_poly = poly.transform(X_range.reshape(-1,1))
    y_range_pred = model.predict(X_range_poly)
    
    plt.scatter(X_scaled, y)
    plt.plot(X_range, y_range_pred)
    plt.title(f"Polynomial Regression Degree {d}")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.show()

# Ridge Regression
print("\nRidge Polynomial (Degree 4)")

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_scaled)

ridge = Ridge(alpha=1)
ridge.fit(X_poly, y)

y_ridge = ridge.predict(X_poly)

plt.scatter(X_scaled, y)
plt.plot(X_scaled, y_ridge)
plt.title("Ridge Regularized Polynomial")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()
