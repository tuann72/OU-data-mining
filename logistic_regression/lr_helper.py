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




# sigmoid function for logistic regression model
def sigmoid(z):
    denominator = (1 + np.exp(-z))
    return 1 / denominator

# Train logistic regression model using gradient descent
def train_logistic_regression(data, learning_rate, iterations):
    X = data[:, :-1]
    y = data[:, -1]
    m, n = X.shape

    #initialize empty array for weights and biases to 0
    weights = np.zeros(n)
    biases = 0.0

    # for each iteration update weights and biases
    for iter in range(iterations):
        # dot product of X and weights add biases
        z = np.dot(X, weights) + biases
        # use sigmoid function to make a label prediction
        y_prediction = sigmoid(z)


        error = y_prediction - y

        # Compute gradients
        gradient_weight = (X.T @ error) / m
        gradient_bias = np.mean(error)

        # Update weights and biases using gradients
        weights -= learning_rate * gradient_weight
        biases  -= learning_rate * gradient_bias

        # Every 100 iterations calculate and print loss
        if iter % 100 == 0:
            # use small constant to avoid log(0)
            epsilon = 1e-8

            # Calculate cross-entropy loss
            positive_class_loss = y * np.log(y_prediction + epsilon)
            negative_class_loss = (1 - y) * np.log(1 - y_prediction + epsilon)

            # Calculate the average loss across all examples
            loss = -np.mean(positive_class_loss + negative_class_loss)

            # Print loss
            print(f"Iteration number: {iter:3d}  Loss: {loss: .3f}")

    return weights, biases

def make_prediction(weights, bias, instance):
    z = np.dot(instance, weights) + bias
    if sigmoid(z) >= 0.5:
        return 1
    else:
        return 0

def predict_probability(weights, bias, instance):
    z = np.dot(instance, weights) + bias
    return sigmoid(z)

# Accuracy calculation
def get_accuracy(weights, bias, test_set):
    correct = 0
    for example in test_set:
        # get features
        x = example[:-1]
        # get label
        y_true = int(example[-1])
        y_prediction = make_prediction(weights, bias, x)
        # see if prediction is correct
        if y_prediction == y_true:
            correct += 1
    return correct / len(test_set)


# Calculate precision recall curve
def calculate_precision_recall_curve(weights, bias, test_set):
    y_confidence = []
    y_true = []

    # get predictions and true labels for each example
    for example in test_set:
        true_label = int(example[-1])
        prediction = predict_probability(weights, bias, example[:-1])
        y_confidence.append(prediction)
        y_true.append(true_label)

    y_confidence = np.array(y_confidence)
    y_true = np.array(y_true)

    # sort by confidence 
    sorted_indices = np.argsort(-y_confidence)
    y_true = y_true[sorted_indices]
    y_confidence = y_confidence[sorted_indices]

    # initialize values 
    true_positive, false_positive = 0, 0
    false_negative = np.sum(y_true)
    precision, recall = [], []

   
    for i in range(len(y_true)):
        # update tp fn and fp counts
        if y_true[i] == 1:
            true_positive += 1
            false_negative -= 1
        else:
            false_positive += 1

        # calculate precision and recall avoiding division of 0
        if (true_positive + false_positive) > 0:
            curr_prec = true_positive / (true_positive + false_positive)
        else:
            curr_prec = 1.0
        
        if (true_positive + false_negative) > 0:
            curr_rec = true_positive / (true_positive + false_negative) 
        else:
            curr_rec = 0.0

        precision.append(curr_prec)
        recall.append(curr_rec)
    # Calculate area under curve
    auc = np.trapezoid(precision, recall)
    return precision, recall, auc


# Calculate ROC curve
def calculate_roc_curve(weights, bias, test_set):
    y_confidence = []
    y_true = []


    # get predictions and true labels for each example
    for instance in test_set:
        true_label = int(instance[-1])
        confidence = predict_probability(weights, bias, instance[:-1])
        y_confidence.append(confidence)
        y_true.append(true_label)

    y_confidence = np.array(y_confidence)
    y_true = np.array(y_true)
    # sort by confidence
    sorted_indices = np.argsort(-y_confidence)
    y_true = y_true[sorted_indices]
    y_confidence = y_confidence[sorted_indices]

    # initialize values
    true_positive = 0
    false_positive = 0
    false_negative = np.sum(y_true)
    true_negative = len(y_true) - false_negative

    tpr_list = []
    fpr_list = []

    # Update counts for tp, fn, fp, and tn
    for i in range(len(y_true)):
        if y_true[i] == 1:
            true_positive += 1
            false_negative -= 1
        else:
            false_positive += 1
            true_negative -= 1

        # calculate true positive rate and false positive rate avoiding division by 0
        if (true_positive + false_negative) > 0:
            tpr = true_positive / (true_positive + false_negative) 
        else:
            tpr = 0.0
        
        if (false_positive + true_positive) > 0:
            fpr = false_positive / (false_positive + true_negative) 
        else:
            fpr = 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
    # Calculate area under curve
    auc = np.trapezoid(tpr_list, fpr_list)
    return fpr_list, tpr_list, auc



# calculate confusion matrix
def calculate_confusion_matrix(weights, bias, test_set):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for example in test_set:
        x = example[:-1]
        y_true = int(example[-1])
        y_pred = make_prediction(weights, bias, x)

        if y_true == 1 and y_pred == 1:
            true_positive += 1
        elif y_true == 1 and y_pred == 0:
            false_negative += 1
        elif y_true == 0 and y_pred == 1:
            false_positive += 1
        elif y_true == 0 and y_pred == 0:
            true_negative += 1

    return np.array([[true_positive, false_negative],
                     [false_positive, true_negative]])