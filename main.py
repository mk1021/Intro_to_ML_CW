import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from decision_tree import decision_tree_learning, predict
# from cross_validation import train_test_k_fold
from plotting import generate_tree
from evaluate import evaluate

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

# Predict the labels
clean_prediction = predict(clean_decision_tree, test_clean[:, :-1])
clean_gold = test_clean[:, -1]

noisy_prediction = predict(noisy_decision_tree, test_noisy[:, :-1])
noisy_gold = test_noisy[:, -1]

# Evaluation metrics
print(f"===== CLEAN =====")
evaluate(clean_gold, clean_prediction)
print()
print(f"===== NOISY =====")
evaluate(noisy_gold, noisy_prediction)


plt.show()