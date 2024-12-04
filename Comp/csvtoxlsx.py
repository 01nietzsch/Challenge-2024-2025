# Define the molar fractions and atomic weights of each element in the composition
import pandas as pd
# import os
# import subprocess

# # Molar composition
# molar_fractions = {
#     'Fe': 0.623,
#     'C': 0.00854,
#     'Mn': 0.000104,
#     'Si': 0.000203,
#     'Cr': 0.147,
#     'Ni': 0.0000971,
#     'Mo': 0.0179,
#     'V': 0.00515,
#     'N': 0.00163,
#     'Nb': 0.0000614,
#     'Co': 0.188,
#     'W': 0.00729,
#     'Al': 0.000845
# }

# # Atomic weights (g/mol)
# atomic_weights = {
#     'Fe': 55.845,
#     'C': 12.011,
#     'Mn': 54.938,
#     'Si': 28.085,
#     'Cr': 51.996,
#     'Ni': 58.693,
#     'Mo': 95.95,
#     'V': 50.942,
#     'N': 14.007,
#     'Nb': 92.906,
#     'Co': 58.933,
#     'W': 183.84,
#     'Al': 26.982
# }

# # Calculate the molar mass of the mixture
# total_molar_mass = sum(molar_fractions[element] * atomic_weights[element] for element in molar_fractions)

# # Calculate weight percentage for each element
# weight_percentages = {
#     element: (molar_fractions[element] * atomic_weights[element] / total_molar_mass) * 100
#     for element in molar_fractions
# }

# # Create a DataFrame for better readability
# df_weight_percentages = pd.DataFrame.from_dict(weight_percentages, orient='index', columns=['Weight %'])

# import ace_tools as tools; tools.display_dataframe_to_user(name="Weight Percentage Composition", dataframe=df_weight_percentages)
# df_weight_percentages

def parse_composition_string(composition_string):
    elements = []
    values = []
    temp = ''
    i = 0
    while i < len(composition_string):
        if composition_string[i].isalpha():
            if i + 1 < len(composition_string) and composition_string[i + 1].islower():
                elements.append(composition_string[i:i + 2])
                i += 2
            else:
                elements.append(composition_string[i])
                i += 1
            if temp:
                values.append(temp)
                temp = ''
        else:
            temp += composition_string[i]
            i += 1
    if temp:
        values.append(temp)

    readable_format = {elements[i]: float(values[i]) for i in range(len(elements))}
    df_readable_format = pd.DataFrame.from_dict(readable_format, orient='index', columns=['Molar Fraction'])
    return df_readable_format

# Example usage
composition_string = "Fe0.623C0.00854Mn0.000104Si0.000203Cr0.147Ni0.0000971Mo0.0179V0.00515N0.00163Nb0.0000614Co0.188W0.00729Al0.000845"
df_readable_format = parse_composition_string(composition_string)
print("Readable Molar Fractions:")
print(df_readable_format)
