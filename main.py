import numpy as np
import matplotlib.pyplot as plt
from decision_tree import decision_tree_learning

# Define a simple decision tree structure (you should replace this with your actual tree)
decision_tree = de

# Create synthetic data for demonstration
np.random.seed(0)
X = np.random.rand(100, 2)


# Function to predict using the decision tree
def predict_tree(tree, X):
    if "leaf" in tree:
        return tree["leaf"]
    if X[tree["feature_index"]] < tree["threshold"]:
        return predict_tree(tree["left"], X)
    else:
        return predict_tree(tree["right"], X)


# Plot decision boundaries recursively
def plot_decision_boundaries(tree, X, depth=0, max_depth=None):
    if max_depth is not None and depth > max_depth:
        return

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=[predict_tree(tree, x) for x in X],
                cmap='coolwarm', s=20, edgecolor='k')

    if "leaf" in tree:
        plt.title(f"Leaf Node (Class {tree['leaf']})")
    else:
        plt.title(
            f"Internal Node: Feature {tree['feature_index']} < {tree['threshold']}")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    if X.shape[0] > 0:
        split_point = tree["threshold"]
        plt.axvline(split_point, color='gray', linestyle='--', linewidth=2)

        left_mask = X[:, tree["feature_index"]] < split_point
        right_mask = X[:, tree["feature_index"]] >= split_point

        plot_decision_boundaries(tree["left"], X[left_mask], depth + 1,
                                 max_depth)
        plot_decision_boundaries(tree["right"], X[right_mask], depth + 1,
                                 max_depth)


# Call the decision tree plot function
plot_decision_boundaries(decision_tree, X, max_depth=2)

plt.show()
