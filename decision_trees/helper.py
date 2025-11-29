import math
import random 
import numpy as np

class TreeNode:
    def __init__(self, is_leaf=False, label=None, attribute=None, threshold=None, prob=None):
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.threshold = threshold
        self.branches = {}
        self.prob = prob

MIN_INSTANCES = 150
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

def split_data(data):
    m, n = np.shape(data)
    # 70-30 split for training and testing
    num_train = (int) (0.7 * m)
    
    training = []
    visited = set()
    # Gets m unique instances for training
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

def entropy(data):
    pos = sum(row[-1] for row in data)
    probs = [1 - (pos / len(data)), pos / len(data)]
    return -sum(p * math.log(p, 2) for p in probs if p > 0)

def information_gain(data, attribute_index, threshold=None):
    total_impurity = entropy(data)

    if threshold is not None:
        # Continuous attribute
        left = [row for row in data if float(row[attribute_index]) <= threshold]
        right = [row for row in data if float(row[attribute_index]) > threshold]

        if not left or not right:
            return 0

        weighted_impurity = (
            len(left) / len(data) * entropy(left) +
            len(right) / len(data) * entropy(right)
        )

        gain = total_impurity - weighted_impurity
        return gain

    else:
        # Categorical attribute
        subsets = {}
        for row in data:
            key = row[attribute_index]
            subsets.setdefault(key, []).append(row)

        weighted_impurity = sum(
            len(subset) / len(data) * entropy(subset)
            for subset in subsets.values()
        )

        gain = total_impurity - weighted_impurity
        return gain

def optimal_threshold(data, attribute_index):
    # Sort data by the given attribute
    sorted_data = sorted(data, key=lambda row: row[attribute_index])

    best_gain = -float('inf')
    best_threshold = None
    total_impurity = entropy(sorted_data)

    for i in range(len(sorted_data) - 1):
        current_label = sorted_data[i][-1]  
        next_label = sorted_data[i + 1][-1]

        # Find where instances differ in target to determine threshold for continuous attribute
        if current_label != next_label:
            val1 = sorted_data[i][attribute_index]
            val2 = sorted_data[i + 1][attribute_index]
            threshold = (val1 + val2) / 2

            left = [row for row in data if row[attribute_index] <= threshold]
            right = [row for row in data if row[attribute_index] > threshold]

            # Skip threshold candidate if it did not split data
            if not left or not right:
                continue

            weighted_impurity = (
                len(left) / len(data) * entropy(left) +
                len(right) / len(data) * entropy(right)
            )

            gain = total_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

    return best_threshold if best_threshold is not None else None

def majority_class(data):
    # Since target is 0 or 1, sum will count number of positive instances
    return 1 if sum(data[i][-1] for i in range(len(data))) >= (len(data) / 2) else 0

def choose_best_feature(data, attributes, categorical_indices):
    max_gain, ind = -float("inf"), -1
    for attribute in attributes:
        if attribute in categorical_indices:
            gain = information_gain(data, attribute, None)
            if gain > max_gain:
                max_gain = gain
                ind = attribute
        else:
            thresh = optimal_threshold(data, attribute)
            gain = information_gain(data, attribute, thresh)
            if gain > max_gain:
                max_gain = gain
                ind = attribute

    return ind

def construct_tree(data, attributes, categorical_indices):
    # Create Root
    root = TreeNode()

    # If all pos, return single-node tree Root with label +
    target_sum = sum(data[i][-1] for i in range(len(data)))
    if target_sum == len(data):
        root.label = 1
        root.is_leaf = True
        root.prob = 1.0
        return root

    # If all neg, return single-node tree Root with label -
    if target_sum == 0:
        root.label = 0
        root.is_leaf = True
        root.prob = 0.0
        return root

    # If attributes empty/too few instances, return single-node tree Root w label = majority class
    if not attributes or len(data) < MIN_INSTANCES: 
        root.label = majority_class(data)
        root.is_leaf = True
        root.prob = target_sum / len(data)
        return root

    # A = best attribute from data (highest info gain)
    A = choose_best_feature(data, attributes, categorical_indices)
    new_attributes = attributes.copy()
    new_attributes.remove(A)
    
    # Decision attribute for Root = A
    root.attribute = A

    if A in categorical_indices: 
        #values = set(row[A] for row in data)
        values = possible_values[A]

        # For v in A (iterate through all possible values of A)
        for v in values:
        
            # Add new branch below Root corresponding to A = v

            # data_v = subset of data that have A = v
            data_v = [row for row in data if row[A] == v]
            
            # If not data_v (data_v is empty)
            if not data_v:
                # Create leaf node w label = majority class
                leaf = TreeNode(is_leaf=True, label=majority_class(data), prob=target_sum/len(data))
                root.branches[v] = leaf
            else: 
                # Create subtree id3(data_v, target, attributes - {A})
                subtree = construct_tree(data_v, new_attributes, categorical_indices)
                root.branches[v] = subtree
        # Return root
        return root
    else:
        root.threshold = optimal_threshold(data, A)
        threshold = root.threshold
        # If a threshold cannot be found for the attribute, make it a leaf
        if not threshold: 
            root.is_leaf = True
            root.label = majority_class(data)
            root.prob = target_sum / len(data)
        else:
            left = [row for row in data if row[A] <= threshold]
            right = [row for row in data if row[A] > threshold]

            root.branches[0] = construct_tree(left, new_attributes, categorical_indices) if left else TreeNode(is_leaf=True, label=majority_class(data))
            root.branches[1] = construct_tree(right, new_attributes, categorical_indices) if right else TreeNode(is_leaf=True, label=majority_class(data))
        return root

def predict(root, query):
    while not root.is_leaf:
        A = root.attribute
        if root.threshold is not None:
            if query[A] <= root.threshold:
                #print(f"Att {A}, Query {query[A]} <= Thresh {root.threshold} ")
                return predict(root.branches[0], query)
            else:
                #print(f"Att {A}, Query {query[A]} > Thresh {root.threshold} ")
                return predict(root.branches[1], query)
        else:
            #print(f"Att {A}, Query {query[A]}")
            if query[A] not in root.branches:
                print(A)
                print(root.branches)
            return predict(root.branches[query[A]], query)
    return root.label

def accuracy(tree, test_set):
    correct = 0
    for instance in test_set:
        if predict(tree, instance) == int(instance[-1]):
            correct += 1
    return correct / len(test_set)

def compute_pr_curve(tree, test_set):
    y_scores = []
    y_true = []

    for instance in test_set:
        true_label = int(instance[-1])
        predicted_label = predict(tree, instance)
        score = predicted_label 

        y_scores.append(score)
        y_true.append(true_label)

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    y_scores = y_scores[sorted_indices]

    tp, fp = 0, 0
    fn = np.sum(y_true)
    precision, recall = [], []

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

    auc = np.trapezoid(precision, recall)
    return precision, recall, auc

def predict_proba(root, query):
    node = root
    while not node.is_leaf:
        A = node.attribute
        if node.threshold is not None:
            if query[A] <= node.threshold:
                node = node.branches[0]
            else:
                node = node.branches[1]
        else:
            if query[A] in node.branches:
                node = node.branches[query[A]]
            else:
                # Default to 50-50 if value was unseen
                return 0.5
    # Estimate probability as proportion of positive labels at the leaf
    return node.prob

def compute_roc_curve(tree, test_set):
    y_scores = []
    y_true = []

    for instance in test_set:
        true_label = int(instance[-1])
        score = predict_proba(tree, instance)
        y_scores.append(score)
        y_true.append(true_label)

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    # Sort scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true = y_true[sorted_indices]
    y_scores = y_scores[sorted_indices]

    tp = 0
    fp = 0
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
