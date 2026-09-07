import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Part 1: Regression Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("REGRESSION METRICS")
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# Part 2: K-Fold Cross Validation
scores_5 = cross_val_score(model, X, y, cv=5, scoring="r2")
scores_10 = cross_val_score(model, X, y, cv=10, scoring="r2")

print("\nK-FOLD CROSS-VALIDATION")
print("K=5 scores:", scores_5)
print("K=5 average R²:", scores_5.mean())

print("\nK=10 scores:", scores_10)
print("K=10 average R²:", scores_10.mean())

# Interpretation
if scores_10.std() < scores_5.std():
    print("\nInterpretation: K=10 gave more consistent results.")
else:
    print("\nInterpretation: K=5 gave more consistent results.")