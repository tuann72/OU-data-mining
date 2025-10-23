import numpy as np
from helper import *
import matplotlib.pyplot as plt

# Load data and split into training and testing sets
# skip first row cause of column headers
data = np.loadtxt('../clean_data.csv', delimiter=',', skiprows=1)
training_set, testing_set = split_data(data)

# Train logistic regression model
updated_array = np.array(training_set)
model_weights, model_biases = train_logistic_regression(updated_array, 0.05,1000)

# get model accuracy
model_accuracy = get_accuracy(model_weights, model_biases, np.array(testing_set))
print(f"Model Accuracy: {model_accuracy: .3f}")


# Calculate and plot precision recall curve
model_precision, model_recall, auc_pr = calculate_precision_recall_curve(model_weights, model_biases, testing_set)
plt.plot(model_recall, model_precision, label=f'AUC = {auc_pr: .3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Logistic Regression Precision Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Calulate and plot ROC Curve
false_positive_rate, true_positive_rate, auc_roc = calculate_roc_curve(model_weights, model_biases, testing_set)
plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {auc_roc: .3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

