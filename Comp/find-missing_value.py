import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
path = r"C:\Users\vladimir jurien\OneDrive - Imperial College London\Imperial\Y2\Steel Challenge\Challenge-2024-2025\Comp\\"
# Load the data
data = pd.read_excel(path +'merged_file.xlsx').copy()

# Extract compositional elements
comp_columns = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']

# Identify mechanical properties
mech_properties = ['yield strength', 'tensile strength', 'elongation']



# Function to impute missing values using k-nearest neighbors
def impute_with_knn(df, k=5):
    
    # Create a copy of the dataframe
    imputed_df = df.copy()
    
    
    # Separate compositional and mechanical data
    comp_data = df[comp_columns]
    mech_data = df[mech_properties]
    
    # Calculate Euclidean distances between compositional vectors
    distances = euclidean_distances(comp_data)
    
    # Impute each mechanical property
    for prop in mech_properties:
        # Find rows with missing values for this property
        missing_mask = mech_data[prop].isna()
        
        if missing_mask.any():
            # For each row with missing value
            for idx in mech_data[missing_mask].index:
                # Find k nearest neighbors, excluding the current row itself
                distances_to_current = distances[idx].copy()
                distances_to_current[idx] = np.inf  # Exclude self
                
                # Get indices of k nearest neighbors
                k_nearest_indices = np.argsort(distances_to_current)[:k]
                
                # Get the values of the property for these neighbors
                neighbor_values = mech_data.loc[k_nearest_indices, prop]
                neighbor_values = neighbor_values[~neighbor_values.isna()]
                
                # Impute with weighted average (closer neighbors have higher weight)
                if len(neighbor_values) > 0:
                    # Weights inversely proportional to distance
                    weights = 1 / distances_to_current[k_nearest_indices[:len(neighbor_values)]]
                    weights = weights / weights.sum()
                    
                    imputed_value = np.average(neighbor_values, weights=weights)
                    imputed_df.loc[idx, prop] = imputed_value
    
    return imputed_df

# Perform imputation
imputed_data = impute_with_knn(data, k=5)

# Print summary of imputation
print("Original missing values:")
print(data[mech_properties].isna().sum())
print("\nImputed missing values:")
print(imputed_data[mech_properties].isna().sum())

# Optional: Save imputed dataset
imputed_data.to_excel('imputed_materials_data.xlsx', index=False)

# Visualization of imputation results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i, prop in enumerate(mech_properties):
    plt.subplot(1, 3, i+1)
    plt.scatter(data[prop], imputed_data[prop], alpha=0.7)
    plt.plot([data[prop].min(), data[prop].max()], 
             [data[prop].min(), data[prop].max()], 
             'r--', label='Original = Imputed')
    plt.title(f'Original vs Imputed {prop}')
    plt.xlabel('Original')
    plt.ylabel('Imputed')
    plt.legend()

plt.tight_layout()
plt.show()

# Additional validation
def validate_imputation(original, imputed):
    print("\nImputation Validation:")
    for prop in mech_properties:
        original_mean = original[prop].mean()
        imputed_mean = imputed[prop].mean()
        original_std = original[prop].std()
        imputed_std = imputed[prop].std()
        
        print(f"\n{prop}:")
        print(f"Original Mean: {original_mean:.2f}")
        print(f"Imputed Mean: {imputed_mean:.2f}")
        print(f"Original Std Dev: {original_std:.2f}")
        print(f"Imputed Std Dev: {imputed_std:.2f}")

validate_imputation(data, imputed_data)