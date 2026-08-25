import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace"],
    "Age": [20, 21, 19, 22, 20, 23, 21],
    "Marks": [85.5, 78.0, 92.5, 88.0, 76.5, 90.0, 84.0],
    "City": ["Tirupati", "Guntur", "Vijayawada", "Kadapa", "Nellore", "Chennai", "Hyderabad"]
}

df = pd.DataFrame(data)

print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nShape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)