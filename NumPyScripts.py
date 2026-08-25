import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

print("Original Array:")
print(arr)

sliced_arr = arr[1:5]

print("\nSliced Array (index 1 to 4):")
print(sliced_arr)

reshaped_arr = arr.reshape(2, 3)

print("\nReshaped Array (2 x 3):")
print(reshaped_arr)

broadcasted_arr = reshaped_arr + 10

print("\nAfter Broadcasting (Adding 10):")
print(broadcasted_arr)

row = np.array([1, 2, 3])

result = reshaped_arr + row

print("\nBroadcasting with [1, 2, 3]:")
print(result)