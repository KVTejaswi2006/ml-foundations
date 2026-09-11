import pandas as pd

df = pd.DataFrame({
    "MonthlyCharges": [50, 70, 40, 90, 60],
    "Tenure": [12, 24, 6, 36, 18],
    "Age": [25, 35, 18, 55, 28],
    "Income": [45000, 75000, 25000, 120000, 55000]
})

df["TotalCharges"] = df["MonthlyCharges"] * df["Tenure"]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 50, 100],
    labels=["Teen", "Young", "Middle", "Senior"]
)

df["IncomeCategory"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 100000, float("inf")],
    labels=["Low", "Medium", "High", "Very High"]
)

print(df)