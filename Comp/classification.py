#THIS FILE WILL HAVE STEP 5

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Load the data
path_vlad = r"C:\Users\vladimir jurien\OneDrive - Imperial College London\Imperial\Y2\Steel Challenge\Challenge-2024-2025\final_steel_data.xlsx"
path_damaso = '/Users/damasomatheus/Desktop/Damaso\'s Stuff/Imperial/Materials/Year 2/MATE50001/coding challenge 24/Challenge-2024-2025/final_steel_data.xlsx'
who = input('Who are you? ')
if who == 'vlad':
    path = path_vlad
    
else:
    path = path_damaso
data = pd.read_excel(path)
data = data[data['strength_rating'] != 'Unknown']

#Inputs (standardized compositions)
X = data[['fe', 'c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]

#Outputs (encoded strength rating: fragile, medium, strong)
y = data['strength_rating']

#Split the data into training and testing sets
RANDOM_STATE = 42
TEST_SIZE = 0.2 #80/20 split training/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
) #We have to stratify the data so that the distribution of the target variable is the same in both the training and testing sets e.g., there's only a few fragile samples

#Initialize classifiers
models = {
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced")
}

#Train and evaluate models
for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    unique_labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_reversed = cm[::-1]

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_reversed, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels[::-1])  # Reverse only y-tick labels
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()

# Train and Save Best Classifier
best_classifier = None
best_accuracy = float("-inf")

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = model

# Save the best classifier to a file
joblib.dump(best_classifier, "elongation_classifier.pkl")
print("\nBest classifier saved for elongation!")