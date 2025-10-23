import numpy as np
from helper import *

# Load data and split into training and testing sets
# skip first row cause of column headers
data = np.loadtxt('../clean_data.csv', delimiter=',', skiprows=1)
training_set, testing_set = split_data(data)

# Train logistic regression model
updated_array = np.array(training_set)
model_weights, model_biases = train_logistic_regression(updated_array, 0.05,1000)