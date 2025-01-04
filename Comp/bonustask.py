#THIS FILE WILL HAVE STEP 6

import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained models
tensile_regressor = joblib.load("tensile_strength_regressor.pkl")
yield_regressor = joblib.load("yield_strength_regressor.pkl")
elongation_classifier = joblib.load("elongation_classifier.pkl")

# Define the bounds for each element (in weight %)
bounds = {
    "fe": (50, 95),  # Iron (essential in steel)
    "c": (0, 2),     # Carbon 
    "mn": (0, 10),   # Manganese
    "ni": (0, 5),    # Nickel (minimize)
    "co": (0, 2),    # Cobalt (minimize)
    "cr": (0, 15),   # Chromium
    "mo": (0, 5),    # Molybdenum
    "v": (0, 2),     # Vanadium
    "n": (0, 0.5),   # Nitrogen
    "nb": (0, 2),    # Niobium
    "w": (0, 5),     # Tungsten
    "al": (0, 2),    # Aluminum
    "ti": (0, 2),    # Titanium
    "si": (0, 5),    # Silicon 
}

def generate_compositions(bounds, n_samples=1000):
    compositions = []
    for _ in range(n_samples):
        comp = {element: random.uniform(*bounds[element]) for element in bounds}
        # Scale to ensure the sum is 100%
        total = sum(comp.values())
        for element in comp:
            comp[element] = comp[element] / total * 100
        compositions.append(comp)
    return pd.DataFrame(compositions)

candidates = generate_compositions(bounds, n_samples=1000)

# Reorder columns so that they're the same as the excel file
feature_order = ['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']
candidates = candidates[feature_order]

# Predict outputs
candidates["tensile_strength"] = tensile_regressor.predict(candidates[feature_order])
candidates["yield_strength"] = yield_regressor.predict(candidates[feature_order])
candidates["elongation_class"] = elongation_classifier.predict(candidates[feature_order])

print("Number of candidates before filtering:", len(candidates))
#print(candidates[["tensile_strength", "yield_strength", "elongation_class"]].head(20))
#print(candidates.head(20))

# Filter valid candidates
filtered_candidates = candidates[
    (candidates["tensile_strength"] >= 2000) &
    (candidates["yield_strength"] >= 1500) &
    (candidates["elongation_class"].isin(["Medium", "Strong"])) & 
    (candidates["ni"] <= 2) &
    (candidates["co"] <= 2)
]
print("Number of candidates after filtering:", len(filtered_candidates))

# Select and display the top candidates
optimized_candidates = filtered_candidates.sort_values(["ni", "co"]).head(10)
print("Optimized Candidates:")
print(optimized_candidates)

# Plot Ni vs. Co for optimized candidates
plt.scatter(optimized_candidates["ni"], optimized_candidates["co"], c="blue")
plt.xlabel("Nickel Content (%)")
plt.ylabel("Cobalt Content (%)")
plt.title("Optimized Compositions: Ni vs Co")
plt.show()