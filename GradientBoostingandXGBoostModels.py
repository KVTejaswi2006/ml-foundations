import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

print("XGBoost Before Tuning")

model = models["XGBoost"]
start = time.time()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))
print("Time:", round(time.time() - start, 4))

params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nXGBoost After Tuning")
print("Best Parameters:", grid.best_params_)

pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))

print("\nFinal Model Leaderboard")

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    train_time = time.time() - start

    results.append((
        name,
        accuracy_score(y_test, pred),
        f1_score(y_test, pred),
        train_time
    ))

results.sort(key=lambda x: x[1], reverse=True)

print("Model                 Accuracy    F1 Score    Time")
for r in results:
    print(f"{r[0]:22} {r[1]:.4f}      {r[2]:.4f}      {r[3]:.4f}")

print("\nWinner:", results[0][0])