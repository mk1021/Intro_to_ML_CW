import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from decision_tree import decision_tree_learning, predict
from cross_validation import train_test_k_fold
from plotting import generate_tree
from evaluate import evaluate, confusion_matrix, accuracy


def cross_validation(dataset):
    # Separate data_clean into data and labels
    data = dataset[:, :-1]
    labels = dataset[-1]

    # Computing the accuracy for each fold
    n_folds = 10
    accuracies = np.zeros((n_folds,))
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(data), rg)):
        # Get the dataset from the correct splits
        train_data = data[train_indices, :]
        train_labels = labels[train_indices]
        test_data = data[test_indices, :]
        test_labels = labels[test_indices]

        # Initialize a list to store predictions
        predictions = []

        # Iterate through the test instances and make predictions directly
        decision_tree = decision_tree_learning(np.stack(train_data, train_labels), 0)[0]
        predictions = predict(decision_tree, test_data)
        confusion = confusion_matrix(test_labels, predictions)
        accuracies[i] = accuracy(confusion)
        # print(accuracies)

    return accuracies.mean(), accuracies.std()


""" ------------------------------------ MAIN ---------------------------------------------"""

# Loading the data from the text file
data_clean = np.loadtxt('data/clean_dataset.txt')
data_noisy = np.loadtxt('data/noisy_dataset.txt')

# CLEAN: Split the dataset into training and test
test_total = round(len(data_clean) * 0.2)
rg = default_rng(60012)
rand_indexes = rg.permutation(len(data_clean))
test_clean = data_clean[rand_indexes[:test_total]]
train_clean = data_clean[rand_indexes[test_total:]]

# NOISY: Split the dataset into training and test
test_total = round(len(data_noisy) * 0.2)
rg = default_rng(60012)
rand_indexes = rg.permutation(len(data_noisy))
test_noisy = data_noisy[rand_indexes[:test_total]]
train_noisy = data_noisy[rand_indexes[test_total:]]

# Produce trees for both datasets
clean_decision_tree, clean_max_depth = decision_tree_learning(train_clean, 0)
noisy_decision_tree, noisy_max_depth = decision_tree_learning(train_noisy, 0)

# Print tree for both datasets
generate_tree(clean_decision_tree, clean_max_depth, "clean_tree.png")
generate_tree(noisy_decision_tree, noisy_max_depth, "noisy_tree.png")

# Cross-validation

# CLEAN
accuracy_clean = cross_validation(data_clean)

# NOISY
accuracy_noisy = cross_validation(data_noisy)


# Predict the labels
clean_prediction = predict(clean_decision_tree, test_clean[:, :-1])
clean_gold = test_clean[:, -1]

noisy_prediction = predict(noisy_decision_tree, test_noisy[:, :-1])
noisy_gold = test_noisy[:, -1]

# Evaluation metrics
print(f"===== CLEAN =====")
evaluate(clean_gold, clean_prediction)

print(f"Average Accuracy: \n {accuracy_clean[0]}")
print(f"Accuracy Standard Deviation: \n {accuracy_clean[1]}")
print()
print(f"===== NOISY =====")
evaluate(noisy_gold, noisy_prediction)

print(f"Average Accuracy: \n {accuracy_noisy[0]}")
print(f"Accuracy Standard Deviation: \n {accuracy_noisy[1]}")


plt.show()