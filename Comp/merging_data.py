# Define the molar fractions and atomic weights of each element in the composition
import pandas as pd

def parse_composition_string(composition_string):
    elements = []
    values = []
    temp = ''
    i = 0
    while i < len(composition_string):
        if composition_string[i].isalpha():
            if i + 1 < len(composition_string) and composition_string[i + 1].islower():
                elements.append(composition_string[i:i + 2].lower())
                i += 2
            else:
                elements.append(composition_string[i].lower())
                i += 1
            if temp:
                values.append(temp)
                temp = ''
        else:
            temp += composition_string[i]
            i += 1
    if temp:
        values.append(temp)

    molar_fractions = {elements[i]: float(values[i]) for i in range(len(elements))}

    # Atomic weights (g/mol)
    atomic_weights = {
        'fe': 55.845,
        'c': 12.011,
        'mn': 54.938,
        'si': 28.085,
        'cr': 51.996,
        'ni': 58.693,
        'mo': 95.95,
        'v': 50.942,
        'n': 14.007,
        'nb': 92.906,
        'co': 58.933,
        'w': 183.84,
        'al': 26.982,
        'ti': 47.867
    }

    # Calculate the molar mass of the mixture
    total_molar_mass = sum(molar_fractions[element] * atomic_weights[element] for element in molar_fractions)

    # Calculate weight percentage for each element
    weight_percentages = {
        element: (molar_fractions[element] * atomic_weights[element] / total_molar_mass) * 100
        for element in molar_fractions
    }

    # Create a DataFrame with the composition string and weight percentages
    data = {'Composition String': [composition_string]}
    data.update({element: [weight_percentages[element]] for element in weight_percentages})
    df_weight_percentages = pd.DataFrame(data)
    # drop the column "Fe"
    df_weight_percentages = df_weight_percentages.drop(columns=['fe'])

    return df_weight_percentages

# read a xlsx file
df = pd.read_excel('database_steel_properties.xlsx')


all_weight_percentages = pd.DataFrame()

# Iterate through each composition string in the DataFrame
for string in df['formula']:
    df_weight_percentages = parse_composition_string(string)
    all_weight_percentages = pd.concat([all_weight_percentages, df_weight_percentages], ignore_index=True)

# # save the DataFrame to a new xlsx file
# all_weight_percentages.to_excel('weight_percentages.xlsx', index=False)

# rearrange the columns as follow : Composition String, c,	mn	,si	,cr	,ni	,mo	,v	,n	,nb	,co	,w	,al	,ti
all_weight_percentages = all_weight_percentages[['Composition String', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]

# Read the original file and the new file
df2 = pd.read_excel('weight_percentages2.xlsx')

# Merge the dataframes on 'formula' while prioritizing original file
df3 = df.merge(df2, on='formula', how='left', suffixes=('', '_new'))

# Fill in missing values in the original dataframe with corresponding values from df2
for col in df3.columns:
    if col.endswith('_new'):  # Handle columns from the new file
        original_col = col.replace('_new', '')
        if original_col in df3.columns:  # Fill original column if it exists
            df3[original_col] = df3[original_col].fillna(df3[col])
        else:  # If no matching original column, rename the new column
            df3.rename(columns={col: original_col}, inplace=True)

# Drop '_new' columns if not needed
df3 = df3[[col for col in df3.columns if not col.endswith('_new')]]

# Save the result to a new Excel file
df3.to_excel('merged_file.xlsx', index=False)


