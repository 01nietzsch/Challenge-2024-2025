import pandas as pd
import openpyxl
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_excel('/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/Comp/merged_file.xlsx').copy()
#File already has only input data (metal comps and formula)
inputs = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']] 
scaler = StandardScaler()
normalized_inputs = scaler.fit_transform(inputs)
normalized_data = pd.DataFrame(normalized_inputs, columns=inputs.columns)
output_file = '/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/Comp/normalized_file.xlsx'
normalized_data.to_excel(output_file, index=False)

#verify normalissation worked
print(normalized_data['fe'].mean())
