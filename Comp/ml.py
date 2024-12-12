#THIS FILE WILL HAVE STEPS 4, 5 AND 6

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
file_path = "/path_to_your_combined_file.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)

# Features (standardized compositions)
X = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]

# Targets (unchanged properties)
y_tensile = data['tensile_strength']
y_yield = data['yield_strength']
y_elongation = data['elongation']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_tensile_train, y_tensile_test = train_test_split(X, y_tensile, test_size=0.2, random_state=42)
_, _, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
_, _, y_elongation_train, y_elongation_test = train_test_split(X, y_elongation, test_size=0.2, random_state=42)

# Step 3: Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Step 4: Train and evaluate each model for tensile strength
results = {}
for name, model in models.items():
    model.fit(X_train, y_tensile_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_tensile_test, y_pred)
    mse = mean_squared_error(y_tensile_test, y_pred)
    results[name] = {"R^2": r2, "MSE": mse}

# Print results
print("Results for Tensile Strength:")
for model, metrics in results.items():
    print(f"{model}: R^2 = {metrics['R^2']:.4f}, MSE = {metrics['MSE']:.4f}")