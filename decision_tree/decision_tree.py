import numpy as np


def entropy(data_labels):
    if len(data_labels) == 0:
        return 0
    else:
        unique_labels, label_counts = np.unique(data_labels,
                                                return_counts=True)

        entropy_value = 0.0

        for label in label_counts:
            pk = label / len(data_labels)
            entropy_value += pk * np.log2(pk)

        return -entropy_value


def remainder(left_data_labels, right_data_labels):
    s_left = len(left_data_labels)
    s_right = len(right_data_labels)
    l_fraction = s_left / (s_left + s_right)
    r_fraction = s_right / (s_left + s_right)
    return (l_fraction * entropy(left_data_labels)) + (
            r_fraction * entropy(right_data_labels))


def gain(all_data_labels, left_data_labels, right_data_labels):

    h_all = entropy(all_data_labels)
    r = remainder(left_data_labels, right_data_labels)
    return h_all - r


# function to find the split value
def find_split(data):
    """
        Find the attribute and value that result in the highest information gain for splitting the dataset.

        Args:
            data (numpy.ndarray): The dataset with features and labels.

        Returns:
            tuple: A tuple containing the index of the chosen attribute and the split value.
    """

    col_max_gains = []
    col_max_midvals = []

    # split into cols
    for i in range(len(data[0]) - 1):
        current_col = data[:, i]
        # sort column by indexes and then sort the labels
        sorted_col_indexes = np.argsort(current_col)
        label_col = data[sorted_col_indexes, -1]
        sorted_col = current_col[sorted_col_indexes]
        info_gains = []
        mid_vals = []
        # calculate information gain for each possible split
        for j in range(len(sorted_col) - 1):
            mid_val = np.median(sorted_col[j:j + 2])
            mid_vals.append(mid_val)
            g = gain(label_col, label_col[sorted_col < mid_val],
                     label_col[sorted_col >= mid_val])
            info_gains.append(g)
        # find the max IG for each column
        max_i = np.argmax(info_gains)
        col_max_gains.append(info_gains[max_i])
        col_max_midvals.append(mid_vals[max_i])

    # find split based on max IG from all columns
    max_i = np.argmax(col_max_gains)

    return max_i, col_max_midvals[max_i]


def split_dataset(training_dataset, split):
    """
        Split the training dataset into left and right branches based on the specified split condition.

        Args:
            training_dataset (numpy.ndarray): The training dataset with features and labels.
            split (tuple): A tuple containing the split attribute index and the split value.

        Returns:
            tuple: A tuple containing the left branch and right branch of the dataset.
    """

    left_branch = training_dataset[training_dataset[:, split[0]] < split[1]]
    right_branch = training_dataset[training_dataset[:, split[0]] >= split[1]]
    return left_branch, right_branch


def decision_tree_learning(training_dataset, depth):
    """
        Recursively build a decision tree based on the input training dataset.

        Args:
            training_dataset (numpy.ndarray): The training dataset with features and labels.
            depth (int): The current depth of the decision tree.

        Returns:
            tuple: A tuple containing a dictionary representing the decision tree node and the maximum depth.
    """

    if len(set(training_dataset[:, -1])) == 1:
        # Checking if all samples have the same label
        node = {'leaf': training_dataset[0, -1]}

        return node, depth
        # Returning a leaf node with the label and depth

    else:
        split = find_split(training_dataset)  # Finding the split value
        node = {}  # Creating a new decision tree with root as the split value

        # Splitting the dataset
        l_dataset, r_dataset = split_dataset(training_dataset, split)

        # Recursive call for the left branch
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)

        # Recursive call for the right branch
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

        node['split'] = split
        node['left'] = l_branch
        node['right'] = r_branch

        return node, max(l_depth, r_depth)


def predict(tree, x):
    """
        Predict target values for a given set of instances using a decision tree.

        Args:
            tree (dict): The decision tree model used for prediction.
            x (numpy.ndarray): An array containing the instances to predict.

        Returns:
            numpy.ndarray: An array of predicted target values for the input instances.
    """

    predictions = []
    for instance in x:
        branch = tree
        while 'leaf' not in branch:
            attribute, split_value = branch['split']
            if instance[attribute] < split_value:
                branch = branch['left']
            else:
                branch = branch['right']

        predictions.append(branch['leaf'])

    return np.array(predictions)


