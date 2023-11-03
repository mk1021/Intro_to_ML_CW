import numpy as np
# from matplotlib import pyplot as plt
from numpy.random import default_rng

from decision_tree import decision_tree_learning, predict
from cross_validation import train_test_k_fold
from plotting import generate_tree
from evaluate import evaluate, confusion_matrix, accuracy


def cross_validation(dataset):
    # Computing the prediction for each fold
    n_folds = 10
    fold_confusions = []
    for i, (train_indices, test_indices) in enumerate(
            train_test_k_fold(n_folds, len(dataset), default_rng())):
        # Get the dataset from the correct splits
        train = dataset[train_indices]
        test = dataset[test_indices]

        # Iterate through the test instances and make predictions directly
        decision_tree = decision_tree_learning(train, 0)[0]
        label_prediction = predict(decision_tree, test[:, :-1])
        fold_confusions.append(confusion_matrix(test[:, -1], label_prediction))

    average_confusion = np.mean(np.array(fold_confusions), axis=0)

    return average_confusion


def main(filename):
    # Loading the data from the text file
    data = np.loadtxt(f'data/{filename}.txt')

    # Produce trees for both datasets
    total_decision_tree, max_depth = decision_tree_learning(data, 0)

    # Print tree
    generate_tree(total_decision_tree, max_depth, f"images/{filename}.png")

    # Cross-validation
    avg_confusion = cross_validation(data)
    print(avg_confusion)

    # Evaluation Metrics
    print(evaluate(avg_confusion))


main('clean_dataset')
