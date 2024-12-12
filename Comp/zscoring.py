import pandas as pd
# import openpyxl
from sklearn.preprocessing import StandardScaler, LabelEncoder

path_vlad = r"C:\Users\vladimir jurien\OneDrive - Imperial College London\Imperial\Y2\Steel Challenge\Challenge-2024-2025\Comp\\"
path_damaso = '/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/Comp/'
who = input('Who are you? ')
if who == 'vlad':
    path = path_vlad
    
else:
    path = path_damaso

data = pd.read_excel(path + "merged_file.xlsx").copy()
#File already has only input data (metal comps and formula)
inputs = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']] 
scaler = StandardScaler()
normalized_inputs = scaler.fit_transform(inputs)
normalized_data = pd.DataFrame(normalized_inputs, columns=inputs.columns)
# i dont want to save the data i keep locally
# output_file = '/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/Comp/normalized_file.xlsx'
# normalized_data.to_excel(output_file, index=False)
print(normalized_data.head())

#verify normalissation worked
print(normalized_data.mean())  # Should be close to 0 for all columns
print(normalized_data.std())  # Should be close to 1 for all columns

#ENCODING (label) - ordinal

#function to classify elongation into categories: fragile, medium and strong
def classify_strength(elongation):
    if elongation < 5:
        return "Fragile"
    elif 5 <= elongation <= 10:
        return "Medium"
    else:
        return "Strong"
    
#TEMPORARY FILE PATH (The merged file doesn't have the properties data)
data_with_properties = pd.read_excel(path + "database_steel_properties.xlsx") 
data_with_properties['strength_rating'] = data_with_properties['elongation'].apply(classify_strength)
output_file_2 = path + "encoded_elongation_file.xlsx" 
data_with_properties.to_excel(output_file_2, index=False)

# Verify the first few rows
print(data_with_properties[['elongation', 'strength_rating']].head())

# Optionally, check the distribution of labels
print(data_with_properties['strength_rating'].value_counts())

#IDEALLY WE MERGE ALL THE FILES FOR STEP 4 (standardized compositions, unchanged properties and extra encoded column)
