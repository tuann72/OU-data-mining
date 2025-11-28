import numpy as np
import matplotlib.pyplot as plt
from helper import *

# Read in data
data = np.loadtxt('../clean_data.csv', delimiter=',', skiprows=1)
# Get number of examples m and dimensionality n
m, n = np.shape(data)

# Split dataset into training and testing
train_set, test_set = split_data(data)

categorical_indices = {i for i in range(n)}
numerical_indices = {13, 15, 16, 17, 18, 19}
categorical_indices = categorical_indices - numerical_indices

tree = construct_tree(train_set, [i for i in range(n-1)], categorical_indices)
# For random forest, make many decision trees, each trained on different subsets of data w/ feature randomness
# and bootstrapping, classification is majority vote of all decision trees

acc = accuracy(tree, test_set)
print(f"Accuracy: {acc}")

# Plot PR Curve
precision_e, recall_e, auc = compute_pr_curve(tree, test_set)
plt.plot(recall_e, precision_e, label=f'AUC = {auc:.4f}')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('PR Curve for Decision Tree', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()

print(f"AUC (PR): {auc}")

fpr, tpr, auc = compute_roc_curve(tree, test_set)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"AUC (ROC): {auc:.4f}")