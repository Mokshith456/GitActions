import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Data
x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [16, 25, 36, 49, 64, 81, 100]

# Step 2 - Fitting Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Step 4 - Linear Regression prediction
print("Linear Regression Prediction for x=11:", lin_reg.predict([[11]]))

# Step 5 - Polynomial Regression (Degree 2 for actual non-linear fit)
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),  # Changed degree to 2 for non-linear regression
    LinearRegression(),
)
polynomial_regression.fit(x, y)
X_height = [[20.0]]
target_predicted = polynomial_regression.predict(X_height)
print("Polynomial Regression Prediction for x=20:", target_predicted)
