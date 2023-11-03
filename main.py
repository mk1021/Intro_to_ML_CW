from decision_tree import decision_tree_learning, predict
from cross_validation import train_test_k_fold
from plotting import generate_tree
from evaluate import

# Loading the data from the text file
data_clean = np.loadtxt('data/clean_dataset.txt')
data_noisy = np.loadtxt('data/noisy_dataset.txt')

# Printing the loaded data
# print(data_clean)

# For quick testing
# print(find_split(data_clean))
# print(split_dataset(data_clean, find_split(data_clean))[1])
# print(decision_tree_learning(data_clean, 0))



# # Call the decision tree plot function
test_total = round(len(data_clean) * 0.2)
rg = default_rng(60012)
rand_indexes = rg.permutation(len(data_noisy))
test = data_clean[rand_indexes[:test_total]]
train = data_clean[rand_indexes[test_total:]]

decision_tree = decision_tree_learning(train, 0)[0]
maxdepth = decision_tree_learning(train, 0)[1]
y_prediction = predict(decision_tree, test[:, :-1])
y_gold = test[:, -1]

print(decision_tree)
print(np.sum(y_gold == y_prediction) / len(y_gold))

generate_tree(decision_tree, maxdepth, "tree.png")

plt.show()