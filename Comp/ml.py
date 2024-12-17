#THIS FILE WILL HAVE STEPS 4, 5 AND 6

import pandas as pd
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

#Inputs (standardized compositions)
X = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]

#Outputs (mechanical properties)
targets = {
    "Tensile Strength": data['tensile strength'],
    "Yield Strength": data['yield strength'],
    "Elongation": data['elongation']
}

#Split data and train models for each output
RANDOM_STATE = 42
TEST_SIZE = 0.2
r2_results = {}
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Decision Tree Regression": DecisionTreeRegressor(),
}

#Test models of each property
for target_name, y in targets.items():
    print(f"\n--- Training models for {target_name} ---")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate R² and MSE
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Print results
        print(f"{model_name}: R² = {r2:.4f}, MSE = {mse:.2f}")

#Plot conclusions (ie random forest is best)
plt.figure(figsize=(10, 6))
plt.bar(r2_results.keys(), r2_results.values(), color='skyblue')
plt.ylabel("R² Score")
plt.title(f"Model Comparison (R²) for {target_name}")
plt.xticks(rotation=45)
plt.show()