import numpy as np
from helper import *

# Read in data
data = np.loadtxt('../clean_data.csv', delimiter=',', skiprows=1)
# Get number of examples m and dimensionality n
m, n = np.shape(data)

# Split dataset into training and testing
train_set, test_set = split_data(data)

categorical_indices = {i for i in range(n)}
numerical_indices = {6, 12, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
categorical_indices = categorical_indices - numerical_indices

tree = construct_tree(train_set, [i for i in range(n-1)], categorical_indices)

acc = accuracy(tree, test_set)
print(acc)