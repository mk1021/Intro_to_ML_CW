import numpy as np

# Loading the data from the text file
data_clean = np.loadtxt('data/clean_dataset.txt')
data_noisy = np.loadtxt('data/noisy_dataset.txt')

# Printing the loaded data
print(data_clean)


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


def entropy(dataset):
    if len(dataset) == 0:
        return 0
    else:
        # class_labels = [row[-1] for row in dataset]
        class_labels = []
        for line in dataset:
            if line.strip() != "":
                row = line.strip().split("\t")
                class_labels.append(row[-1])

        unique_labels, label_counts = np.unique(class_labels,
                                                return_counts=True)

        entropy_value = 0.0

        for label in label_counts:
            pk = label / len(dataset)
            entropy_value = pk * np.log2(pk)

        return -entropy_value


def remainder(left_dataset, right_dataset):
    s_left = len(left_dataset)
    s_right = len(right_dataset)
    fraction = s_left / (s_left + s_right)
    return (fraction * entropy(left_dataset)) + (
                fraction * entropy(right_dataset))


def gain(all_dataset, left_dataset, right_dataset):
    h_all = entropy(all_dataset)
    r = remainder(left_dataset, right_dataset)
    return h_all - r


def decision_tree_learning(training_dataset, depth):
    if len(set(training_dataset[:, -1])) == 1:
        # Checking if all samples have the same label
        return training_dataset[0, -1], depth
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

        return (node, max(l_depth, r_depth))


# function to find the split value
def find_split(training_dataset):
    # chooses attribute and value that results in the highest information gain

    data = np.array(training_dataset)
    col_max_gains = []

    # split into cols
    for i in range(len(data)-1):
        current_col = data[:, i]
        sorted_col = np.sort(current_col)
        info_gains = []
        mid_val = []
        for j in range(len(sorted_col-1)):
            mid_val.append(np.median(sorted_col[j:j+2]))
            g = gain(sorted_col, sorted_col[:j], sorted_col[j:])
            info_gains.append(g)


    # sort

    # calculate information gain

    # split based on IG

    pass


# function to split the dataset
def split_dataset(training_dataset, split):
    # Implement your logic here
    pass


def confusion_matrix(true_labels, predicted_labels):
    # HINT: you should get a single 4x4 matrix

    # [ TP  FN  |  FP  TN ] ?

    class_labels = np.unique(np.concatenate((true_labels, predicted_labels)), dtype=np.int)
    confusion = np.zeros()

    for i in range(len(true_labels)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]

        # Find indices in class_labels for both the true and predicted labels
        true_index = np.where(class_labels == true_label)[0]
        predicted_index = np.where(class_labels == predicted_label)[0]

        # Update the corresponding cell in the confusion matrix
        confusion[true_index, predicted_index] += 1

    return confusion


def accuracy(confusion):
    # (TP + TN)/(TP + TN + FP + FN)
    # HINT: you can derive the metrics directly from the previously computed confusion matrix

    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0


def precision_rate(true_labels, predicted_labels):
    # (TP)/(TP + FP)

    unique_class_labels = np.unique(np.concatenate(true_labels, predicted_labels))
    precision = np.zeros(len(unique_class_labels))

    for (c, label) in enumerate(unique_class_labels):
        tp = np.sum((true_labels == label) & (predicted_labels == label))
        fp = np.sum((true_labels != label) & (predicted_labels == label))

        if (tp + fp) > 0:
            precision[c] = tp / (tp + fp)

    return precision


def recall_rate(true_labels, predicted_labels):
    # (TP)/(TP + FN)
    # recall rate per class using the eq

    unique_class_labels = np.unique(np.concatenate(true_labels, predicted_labels))
    recall = np.zeros(len(unique_class_labels))

    for (c, label) in enumerate(unique_class_labels):
        tp = np.sum((true_labels == label) & (predicted_labels == label))
        fn = np.sum((true_labels == label) & (predicted_labels != label))

        if (tp + fn) > 0:
            recall[c] = tp / (tp + fn)
        else:
            recall[c] = 0.0

    return recall


def f1_score(precisions, recalls):
    # 2(precision)(recall)/(precision + recall)

    # precisions = precision_rate(true_labels, predicted_labels)
    # recalls = recall_rate(true_labels, predicted_labels)

    assert len(precisions) == len(recalls)

    f = np.zeros(len(precisions), )

    for i in range(len(precisions)):
        current_p = precisions[i]
        current_r = recalls[i]

        if (current_p + current_r) > 0.0:
            f_one = (2*current_p*current_r)/(current_p + current_r)
        else:
            f_one = 0.0

        f[i] = f_one

    return f


# def f1_score(true_labels, predicted_labels):
#     # 2(precision)(recall)/(precision + recall)
#
#     unique_class_labels = np.unique(np.concatenate(true_labels, predicted_labels))
#     precision = np.zeros(len(unique_class_labels))
#     recall = np.zeros(len(unique_class_labels))
#
#     for (c, label) in enumerate(unique_class_labels):
#         tp = np.sum((true_labels == label) & (predicted_labels == label))
#         fp = np.sum((true_labels != label) & (predicted_labels == label))
#         fn = np.sum((true_labels == label) & (predicted_labels != label))
#
#         # PRECISION
#         if (tp + fp) > 0:
#             precision[c] = tp / (tp + fp)
#         else:
#             precision[c] = 0.0
#
#         # RECALL
#         if (tp + fn) > 0:
#             recall[c] = tp / (tp + fn)
#         else:
#             recall[c] = 0.0
#
#     # F1 SCORE
#     assert len(precision) == len(recall)
#     f = np.zeros(len(precision), )
#
#     for i in range(len(precision)):
#         current_p = precision[i]
#         current_r = recall[i]
#         if (current_p + current_r) > 0.0:
#             f_one = (2*current_p*current_r)/(current_p + current_r)
#         else:
#             f_one = 0.0
#
#         f[i] = f_one
#
#     return f


