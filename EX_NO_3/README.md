EXPT NO: 3
Regression and Optimization
Multilinear Regression & Polynomial Regression

Course: Machine Learning Laboratory
Student Name: Jeevanantham K.
Roll Number: 24BAD047
Semester: IV
Date: 04.02.2026

Aim

To implement and analyze:

Multilinear Regression for predicting student academic performance.

Polynomial Regression for predicting vehicle fuel efficiency.
Also, to optimize models using regularization techniques such as Ridge and Lasso.

Software and Libraries Used

Python 3.13

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Scenario 1 – Multilinear Regression
Problem Statement

Predict student academic performance based on multiple academic and behavioral factors.

Dataset

Kaggle: Student Performance in Exams

Input Features

Gender

Parental level of education

Test preparation course

Lunch type

Race/Ethnicity

Target Variable

Final Exam Score = Average of:

Math score

Reading score

Writing score

Steps Performed

Imported required libraries.

Loaded the dataset using Pandas.

Encoded categorical variables using Label Encoding.

Created a new target variable (final_score).

Handled missing values using mean imputation.

Applied feature scaling using StandardScaler.

Split the dataset into training and testing sets.

Trained a Multilinear Regression model.

Predicted student performance.

Evaluated using MSE, RMSE, and R² score.

Interpreted regression coefficients.

Applied Ridge and Lasso for optimization.

Visualizations

Actual vs Predicted scores

Feature coefficient comparison

Residual distribution plot

Scenario 2 – Polynomial Regression
Problem Statement

Predict vehicle fuel efficiency (MPG) based on engine horsepower, where the relationship is non-linear.

Dataset

Kaggle: Auto MPG Dataset

Input Feature

Horsepower

Target Variable

Miles Per Gallon (MPG)

Steps Performed

Loaded the dataset.

Converted horsepower to numeric.

Handled missing values using numeric mean.

Applied feature scaling.

Generated polynomial features of degree 2, 3, and 4.

Trained polynomial regression models.

Evaluated models using MSE, RMSE, and R².

Compared performance across degrees.

Applied Ridge regression to reduce overfitting.

Visualizations

Polynomial curve fitting

Training vs testing error comparison

Overfitting and underfitting demonstration

Results and Observations
Multilinear Regression

Multiple factors influence student performance.

Regularization improved model stability.

Ridge and Lasso reduced coefficient magnitude.

Polynomial Regression

Higher-degree polynomials fit complex patterns.

Degree 4 showed signs of overfitting.

Ridge regression smoothed the curve.

Conclusion

This experiment demonstrated how:

Multilinear regression handles multiple predictors.

Polynomial regression models non-linear relationships.

Regularization prevents overfitting.

Cross-validation and evaluation metrics help in selecting optimal models.

Both models are highly useful in real-world domains such as:

Education analytics

Transportation and automotive engineering
