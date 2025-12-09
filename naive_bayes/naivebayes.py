import numpy as np
import matplotlib.pyplot as plt
from naive_bayes.nb_helper import *
import pandas as pd

# Read in data
data = np.loadtxt('../clean_data.csv', delimiter=',', skiprows=1)
# Get number of examples m and dimensionality n (last column is target so decrement n)
m, n = np.shape(data)
n -= 1

# Split dataset into training and testing
# train_set, test_set = split_data(data)
train_set, test_set = np.loadtxt('../train.csv', delimiter=',', skiprows=0), np.loadtxt('../test.csv', delimiter=',', skiprows=0)

# Initialize values for number of negative and positive cases (in train_set) and mu values for neg/pos
num_neg, num_pos = 0, 0
mu_neg, mu_pos = [0 for i in range(n)], [0 for i in range(n)]

# Mu values for continuous features
for instance in train_set:
    if instance[n] > 0: 
        num_pos += 1
        for i in range(n):
            mu_pos[i] += instance[i]
    else:
        num_neg += 1
        for i in range(n):
            mu_neg[i] += instance[i]

for i in range(n):
    mu_neg[i] = mu_neg[i] / num_neg
    mu_pos[i] = mu_pos[i] / num_pos

# Sigma squared values for continuous features
sigma_neg, sigma_pos = [0 for i in range(n)], [0 for i in range(n)]

for instance in train_set:
    if instance[n] > 0: 
        for i in range(n):
            sigma_pos[i] += (instance[i] - mu_pos[i])**2
    else:
        for i in range(n):
            sigma_neg[i] += (instance[i] - mu_neg[i])**2

# Conditional probs for categorical features
for i in range(n):
    sigma_neg[i] = sigma_neg[i] / (num_neg-1)
    sigma_pos[i] = sigma_pos[i] / (num_pos-1)

categorical_indices = {i for i in range(n)}
numerical_indices = {13, 15, 16, 17, 18, 19}
categorical_indices = categorical_indices - numerical_indices

cat_counts_neg = [{} for i in range(n)]
cat_counts_pos = [{} for i in range(n)]

possible_values = {
    0: [1, 2, 3, 4, 5, 6], # Marital Status
    1: [0, 1], # Daytime/Evening Attendance
    2: [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109], # Nationality
    3: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], # Mothers Qualification
    4: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], # Fathers Qualification
    5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194], # Mothers Occupation
    6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195], # Fathers Occupation
    7: [0, 1], # Displaced
    8: [0, 1], # Educational Special Needs
    9: [0, 1], # Debtor
    10: [0, 1], # Tuition Fees Up to Date
    11: [0, 1], # Gender
    12: [0, 1], # Scholarship Holder
    14: [0, 1], # International
    20: [0, 1] # Target
}

for i in categorical_indices:
    for val in possible_values[i]:
        cat_counts_pos[i][val] = 0
        cat_counts_neg[i][val] = 0

for instance in train_set:
    cls = 'pos' if instance[n] > 0 else 'neg'
    for i in categorical_indices:
        val = instance[i]
        # count occurrences for categorical
        d = cat_counts_pos[i] if cls == 'pos' else cat_counts_neg[i]
        d[val] = d.get(val, 0) + 1

for i in categorical_indices:
    K = len(possible_values[i])
    for val in possible_values[i]:
        cat_counts_pos[i][val] = (cat_counts_pos[i][val] + 1) / (num_pos + K)
        cat_counts_neg[i][val] = (cat_counts_neg[i][val] + 1) / (num_neg + K)


# Calculating precision and recall for PR curve
y_scores = []
y_true = []

# Confusion matrix values
true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0

for instance in test_set:
    # Reset negative and positive class probabilities for each instance
    neg_prior, pos_prior = (num_neg / len(train_set)), (num_pos / len(train_set))
    prob_neg, prob_pos, prediction = predict_naive(categorical_indices, n, pos_prior, neg_prior, instance, mu_neg, mu_pos, sigma_neg, sigma_pos, cat_counts_neg, cat_counts_pos)
    if instance[-1] == 0: 
        if prediction == 0: 
            true_neg += 1
        else:
            false_pos += 1
    else:
        if prediction == 0:
            false_neg += 1
        else:
            true_pos += 1

    score = prob_pos / (prob_pos + prob_neg) 
    y_scores.append(score)
    y_true.append(1 if instance[n] > 0 else 0)

# Get confusion matrix
confusion_matrix = np.array([[true_pos, false_neg],
                     [false_pos, true_neg]])

# Export confusion matrix to csv
pd.DataFrame(confusion_matrix).to_csv("../naivebayes_cm.csv", index=False, header=False)

y_scores = np.array(y_scores)
y_true = np.array(y_true)

sorted_indices = np.argsort(-y_scores)
y_true = y_true[sorted_indices]
y_scores = y_scores[sorted_indices]

tp, fp = 0, 0
fn = np.sum(y_true)
precision, recall = [], []

# Construct precision and recall arrays for PR curve
for i in range(len(y_true)):
    if y_true[i] == 1:
        tp += 1
        fn -= 1
    else:
        fp += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precision.append(prec)
    recall.append(rec)

auc = pr_auc(recall, precision)
print(f"PR-AUC: {auc:.4f}")
accuracy = (true_pos + true_neg) / len(test_set)

# Export precision-recall data to csv
df = pd.DataFrame({
  "m_pre" : precision,
  "m_rec" : recall,
  "auc_pr" : auc
})
df.to_csv("../naivebayes_pr.csv", index=False)

plt.plot(recall, precision, label=f'AUC = {auc:.4f}')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('PR Curve For Gaussian Naive Bayes', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()

fpr, tpr, roc_auc = compute_roc_curve_nb(y_true, y_scores)
print(f"ROC: {roc_auc:.4f}")

# Export ROC data to csv
df = pd.DataFrame({
  "fp" : fpr,
  "tp" : tpr,
  "auc_roc" : roc_auc
})
df.to_csv("../naivebayes_roc.csv", index=False)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve for Gaussian Naive Bayes', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()

print(accuracy * 100)