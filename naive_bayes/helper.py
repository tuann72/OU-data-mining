import numpy as np
import math
import random

def split_data(data):
    m, n = np.shape(data)
    num_train = (int) (0.7 * m)
    
    training = []
    visited = set()
    while len(training) < num_train:
        i = random.randint(0, m - 1)
        while i in visited:
            i = random.randint(0, m - 1)
        visited.add(i)
        training.append((data[i]))

    testing = []
    for i in range(m):
        if i not in visited:
            testing.append(data[i])
    return training, testing

def conditional_prob(attr, mu, sigma_square):
    return (1/math.sqrt(2 * math.pi * sigma_square)) * math.exp(-(((attr - mu)**2) / (2 * sigma_square)))

def pr_auc(recall, precision):
    return np.trapezoid(precision, recall)

def compute_roc_curve_nb(y_true, y_scores):
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    y_scores = y_scores[sorted_indices]

    tp, fp = 0, 0
    fn = np.sum(y_true)
    tn = len(y_true) - fn

    tpr_list = []
    fpr_list = []

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = np.trapezoid(tpr_list, fpr_list)
    return fpr_list, tpr_list, auc