import numpy as np

# Loading the data from the text file
data = np.loadtxt('clean_dataset.txt')
data = np.loadtxt('noisy_dataset.txt')

# Printing the loaded data
print(data)

def entropy(dataset):
    if len(dataset) == 0:
        return 0
    else:
        class_labels = []
        for line in dataset:
            if line.strip() != "":
                row = line.strip().split(" ")
                class_labels.append(row[-1])

        unique_labels, label_counts = np.unique(class_labels, return_counts=True)

        entropy_value = 0.0

        for label in label_counts:
            pk = label / len(dataset)
            entropy_value = pk * np.log2(pk)

        return -entropy_value


def remainder(left_dataset, right_dataset):
    s_left = len(left_dataset)
    s_right = len(right_dataset)
    fraction = s_left / (s_left + s_right)
    return (fraction * entropy(left_dataset)) + (fraction * entropy(right_dataset))


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
    pass


# function to split the dataset
def split_dataset(training_dataset, split):
    # Implement your logic here
    pass
