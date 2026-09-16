import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = {
    "Age": [22,25,28,31,35,40,45,50,23,27,30,34,38,42,47,52,24,29,33,37,41,46,49,55,21,26,32,36,43,48],
    "AnnualSalary": [25000,32000,40000,45000,52000,60000,70000,85000,28000,38000,42000,50000,58000,65000,75000,90000,30000,41000,48000,55000,62000,72000,82000,95000,22000,35000,46000,53000,68000,80000],
    "Experience": [1,2,3,5,6,8,10,12,1,3,4,6,7,9,11,14,2,4,5,7,8,10,13,15,1,2,5,6,9,12],
    "CreditScore": [580,610,640,660,680,700,720,750,590,630,650,670,690,710,730,760,600,645,665,685,705,725,740,780,570,620,655,675,715,735],
    "Purchased": [0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Age", "AnnualSalary", "Experience", "CreditScore"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(model, Xtr, Xte):
    start = time.time()
    model.fit(Xtr, y_train)
    training_time = time.time() - start

    pred = model.predict(Xte)

    return [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0),
        training_time
    ]

knn_before = evaluate(KNeighborsClassifier(n_neighbors=5), X_train, X_test)
knn_after = evaluate(KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled)

svm_before = evaluate(SVC(kernel="rbf"), X_train, X_test)
svm_after = evaluate(SVC(kernel="rbf"), X_train_scaled, X_test_scaled)

print("\nPART 1: BEFORE AND AFTER SCALING")
print("Model   Before   After")
print("KNN     {:.2f}     {:.2f}".format(knn_before[0], knn_after[0]))
print("SVM     {:.2f}     {:.2f}".format(svm_before[0], svm_after[0]))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf")
}

results = []

for name, model in models.items():
    scores = evaluate(model, X_train_scaled, X_test_scaled)
    results.append([name] + scores)

table = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training Time"]
)

print("\nPART 2: MODEL COMPARISON")
print(table.to_string(index=False))

print("\nCONCLUSION")
print("Feature scaling is important for KNN and SVM.")
print("Decision Tree and Random Forest usually do not require scaling.")
print("KNN works using distances between data points.")
print("SVM finds a boundary between different classes.")