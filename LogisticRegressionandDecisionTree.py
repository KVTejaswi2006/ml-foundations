import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Dataset: Customer Churn
data = {
    "Age": [20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58],
    "MonthlyCharges": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Churn": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[["Age", "MonthlyCharges"]]
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- LOGISTIC REGRESSION ----------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

print("LOGISTIC REGRESSION")
print("Predictions:", log_pred)
print("Churn Probabilities:", log_prob)
print("Accuracy:", accuracy_score(y_test, log_pred))

# ---------------- DECISION TREE ----------------
print("\nDECISION TREE")

for depth in [3, 5, 10]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("max_depth =", depth)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

# Visualize Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(12, 7))
plot_tree(
    tree,
    feature_names=["Age", "MonthlyCharges"],
    class_names=["Stay", "Churn"],
    filled=True
)
plt.show()