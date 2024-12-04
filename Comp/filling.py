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
    return df_weight_percentages

parse_composition_string('Fe0.620C0.000953Mn0.000521Si0.00102Cr0.000110Ni0.192Mo0.0176V0.000112Nb0.0000616Co0.146Al0.00318Ti0.0185')
print(parse_composition_string('Fe0.620C0.000953Mn0.000521Si0.00102Cr0.000110Ni0.192Mo0.0176V0.000112Nb0.0000616Co0.146Al0.00318Ti0.0185'))