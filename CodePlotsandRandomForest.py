import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Dataset
data = {
    "Age": [20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58,
            60, 62, 65, 68],
    "BloodPressure": [110, 115, 120, 125, 130, 125, 135, 140, 130, 145,
                      150, 155, 140, 160, 165, 170, 150, 175, 180, 185],
    "Disease": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Age", "BloodPressure"]]
y = df["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("CONFUSION MATRIX")
print(cm)

print("\nCLASSIFICATION METRICS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Disease"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
print("\nFEATURE IMPORTANCE")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", importance)

# Tune n_estimators
trees = [50, 100, 200]
accuracies = []

for n in trees:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, prediction)
    accuracies.append(accuracy)

    print("\nn_estimators =", n)
    print("Accuracy:", accuracy)

# Accuracy vs Number of Trees
plt.plot(trees, accuracies, marker="o")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs n_estimators")
plt.xticks(trees)
plt.show()