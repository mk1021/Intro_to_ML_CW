import numpy as np
from numpy.random import default_rng

# from evaluate import accuracy
# from decision_tree import predict
# from decision_tree import data_clean
# from decision_tree import decision_tree_learning


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    # Generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # Split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

# For quick testing
k_fold_split(10, 2000, rg)

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # Split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]
        
        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])

    return folds


# To test your function (2000 instances, 10 fold)
for (train_indices, test_indices) in train_test_k_fold(10, 2000, rg):
    print("train: ", train_indices)
    print("test: ", test_indices)
    print()


# Sort-of Main for Testing -------------------------------------------------------------------------

# initial stuff
rg = default_rng(60012)

# Separate data_clean into data and labels
data = data_clean[:, :-1]
labels = data_clean[-1]

# Computing the accuracy for each fold
n_folds = 10
accuracies = np.zeros((n_folds, ))
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
    accuracies[i] = accuracy(test_labels, predictions)

# print out the mean accuracy to find the best model?
print(accuracies)
print(accuracies.mean())
print(accuracies.std())
