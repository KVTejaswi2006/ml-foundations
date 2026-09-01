import matplotlib.pyplot as plt
import seaborn as sns

data = [10, 12, 15, 15, 18, 20, 22, 22, 25, 28, 30, 32, 35, 35, 40]

# Histogram
plt.hist(data, bins=5)
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()

# Scatter Plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

plt.scatter(x, y)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Scatter Plot")
plt.show()

# Box Plot
sns.boxplot(y=data)
plt.ylabel("Values")
plt.title("Box Plot")
plt.show()