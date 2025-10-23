import numpy as np
import random

# Split data using 70/30 split
def split_data(data):
    m, n = np.shape(data)
    num_train = int(0.7 * m)

    training_set = []
    visited_set = []  

    # Randomly choose examples for training set
    while len(training_set) < num_train:
        i = random.randint(0, m - 1)
        while i in visited_set:
            i = random.randint(0, m - 1)
        visited_set.append(i)
        training_set.append(data[i])

    # If not in training set add to test set
    testing_set = []
    for i in range(m):
        if i not in visited_set:
            testing_set.append(data[i])

    # return training set and testing set
    return np.array(training_set), np.array(testing_set)

