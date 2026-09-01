import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Age": [18, 20, 22, 25, 28, 30, 32, 35, 40, 45],
    "Salary": [15000, 18000, 22000, 25000, 30000, 35000, 40000, 45000, 50000, 60000],
    "Experience": [0, 1, 2, 3, 5, 6, 8, 10, 12, 15]
}

df = pd.DataFrame(data)

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nStatistics:")
print(df.describe())

# Histogram
plt.hist(df["Salary"], bins=5)
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()

# Scatter Plot
plt.scatter(df["Experience"], df["Salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# Box Plot
sns.boxplot(y=df["Salary"])
plt.ylabel("Salary")
plt.title("Salary Box Plot")
plt.show()

# Insights
print("\n5 Meaningful Insights:")
print("1. Salary generally increases as experience increases.")
print("2. Employees with more experience tend to earn higher salaries.")
print("3. Most salaries are between 15000 and 60000.")
print("4. The salary distribution shows a wider range at higher values.")
print("5. There are no obvious extreme outliers in the salary data.")