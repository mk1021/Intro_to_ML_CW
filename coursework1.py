import numpy as np

# Loading the data from the text file
data = np.loadtxt('clean_dataset.txt')
data = np.loadtxt('noisy_dataset.txt')

# Printing the loaded data
print(data)
 

def decision_tree_learning(training_dataset, depth):
    if len(set(training_dataset[:,-1])) == 1:  # Checking if all samples have the same label
        return (training_dataset[0,-1], depth)  # Returning a leaf node with the label and depth

    else:
        split = find_split(training_dataset)  # Finding the split value
        node = {}  # Creating a new decision tree with root as the split value

        l_dataset, r_dataset = split_dataset(training_dataset, split)  # Splitting the dataset

        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)  # Recursive call for the left branch
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)  # Recursive call for the right branch

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

