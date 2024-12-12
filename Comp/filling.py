import pandas as pd
data = pd.read_excel("imputed_materials_data.xlsx").copy()
#function to classify elongation into categories: fragile, medium and strong
def classify_strength(elongation):
    if elongation < 5:
        return "Fragile"
    elif 5 <= elongation <= 10:
        return "Medium"
    elif elongation > 10 :
        return "Strong"
    else:
        return "Unknown"

data['strength_rating'] = data['elongation'].apply(classify_strength)

output_file = "final_steel_data.xlsx"
data.to_excel(output_file, index=False)
