#THIS FILE WILL HAVE STEP 4

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

#Load the data
path_vlad = r"C:\Users\vladimir jurien\OneDrive - Imperial College London\Imperial\Y2\Steel Challenge\Challenge-2024-2025\Comp\final_steel_data.xlsx\\"
path_damaso = '/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/final_steel_data.xlsx'
who = input('Who are you? ')
if who == 'vlad':
    path = path_vlad
    
else:
    path = path_damaso
data = pd.read_excel(path)
data = data[data['strength_rating'] != 'Unknown'] # Remove rows with unknown strength rating


#Inputs (standardized compositions)
X = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]

#Outputs (mechanical properties)
targets = {
    "Tensile Strength": data['tensile strength'],
    "Yield Strength": data['yield strength'],
    "Elongation": data['elongation']
}

#Define regression models
RANDOM_STATE = 42
TEST_SIZE = 0.2
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=RANDOM_STATE),
}

#Initialize dictionary to store R² results
r2_results = {target_name: {} for target_name in targets.keys()}

#Train models and gather results
for target_name, y in targets.items():
    print(f"\n--- Training models for {target_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_results[target_name][model_name] = r2  # Store R² score
        
        print(f"{model_name}: R² = {r2:.4f}, MSE = {mse:.2f}")

#Plot R² scores for all targets and models
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(models))  # Positions for bars

bar_width = 0.25  # Width of each group of bars

#Plot bars for each target property
for i, (target_name, r2_scores) in enumerate(r2_results.items()):
    r2_values = list(r2_scores.values())
    ax.bar(x + i * bar_width, r2_values, bar_width, label=target_name)

#Customizations
ax.set_xticks(x + bar_width)
ax.set_xticklabels(models.keys(), rotation=45, ha='right')
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison (R² Scores) for All Target Properties")
ax.legend(title="Target Property")

plt.tight_layout()
plt.show()

# Train and Save Best Models
best_models = {}
for target_name, y in targets.items():
    print(f"\n--- Training models for {target_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    best_model = None
    best_r2 = float("-inf")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{model_name}: R² = {r2:.4f}, MSE = {mse:.2f}")
        
        # Save the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Save the best model to a file
    joblib.dump(best_model, f"{target_name.replace(' ', '_').lower()}_regressor.pkl")
    best_models[target_name] = best_model

print("\nBest models saved for each target property!")