import numpy as np
from numpy.random import default_rng

from evaluate import accuracy
from decision_tree import decision_tree_learning


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

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
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

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])

    return folds


# to test your function (2000 instances, 10 fold)
for (train_indices, test_indices) in train_test_k_fold(10, 2000, rg):
    print("train: ", train_indices)
    print("test: ", test_indices)
    print()


def decision_tree_predict(current_node, x):
    # Check if the current node is a leaf node
    trained_dataset = decision_tree_learning(training_dataset, depth)
    current_node = trained_dataset

    if current_node.is_leaf:
        # Determine which child node to follow based on the feature value of the current instance
        feature_index = current_node.feature_index
        feature_value = x[feature_index]

        if feature_value <= current_node.threshold:
            return decision_tree_predict(current_node.left_child, x)
        else:
            return decision_tree_predict(current_node.right_child, x)

    predicted_label = current_node.majority_class

#
# def decision_tree_predict(training_dataset, x, depth):
#     """
#         Predict the class label for an instance using a decision tree.
#
#         Args:
#         tree: The trained decision tree.
#         instance: A numpy array containing the features of the instance.
#
#         Returns:
#         predicted_label: The predicted class label.
#         """
#     # Start at the root of the decision tree
#     trained_dataset = decision_tree_learning(training_dataset, depth)
#     current_node = trained_dataset
#
#     # Traverse the decision tree until we reach a leaf node
#     while not current_node.is_leaf:
#         # Determine which child node to follow based on the feature value of the current instance
#         feature_index = current_node.feature_index
#         feature_value = x[feature_index]
#         if feature_value <= current_node.threshold:
#             current_node = current_node.left_child
#         else:
#             current_node = current_node.right_child
#
#     # The predicted class label is the majority class in the leaf node
#     predicted_label = current_node.majority_class
#     return predicted_label


def retrain(n_folds=10, training_dataset):

    accuracies = np.zeros((n_folds, ))
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x), rg)):
        # get the dataset from the correct splits
        train = training_dataset[train_indices, :]
        test = training_dataset[test_indices, :]

        # DECISION TREE PREDICT
        classifier = decision_tree_learning(train, depth=1)

        # Initialize a list to store predictions
        predictions = []

        # Iterate through the test instances and make predictions directly
        for instance in test:
            # Make a prediction for the current test instance using your classifier & append it to a list of predictions
            predicted_label = decision_tree_predict(train, instance[:-1])  # Assuming the last column is the label
            predictions.append(predicted_label)

        accuracies[i] = accuracy(test[-1], predictions)


    return accuracies.mean()


# print(accuracies)
# print(accuracies.mean())
# we don't need Standard Deviation - print(accuracies.std())
