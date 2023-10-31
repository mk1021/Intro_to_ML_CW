import numpy as np
from numpy.random import default_rng

from evaluate import accuracy


class KNNClassifier:
    def __init__(self, k=10):
        """ K-NN Classifier.

        Args:
            k (int): Number of nearest neighbours. Defaults to 5.
        """
        self.k = k
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x, y):
        """ Fit the training data to the classifier.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)
            y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """ Perform prediction given some examples.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
            y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """ 

        # just to make sure that we have enough training examples!
        k = min([self.k, len(self.x)])

        y = np.zeros((len(x), ), dtype=self.y.dtype)       
        for (i, instance) in enumerate(x):
            distances = np.sqrt(np.sum((instance-self.x)**2, axis=1))
            sorted_indices = np.argsort(distances)
            sorted_indices = sorted_indices[:k]
            
            # Assign to the majority class label (the mode)
            unique_labels, freq = np.unique(self.y[sorted_indices], return_counts=True)
            y[i] = unique_labels[freq.argmax()]

        return y


knn_classifier = KNNClassifier(k=1) # we'll do one nearest neighbour
knn_classifier.fit(x_train, y_train)
knn_predictions = knn_classifier.predict(x_test)
print(knn_predictions)


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

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


# to test your function (30 instances, 4 fold)
for (train_indices, test_indices) in train_test_k_fold(10, 2000, rg):
    print("train: ", train_indices)
    print("test: ", test_indices)
    print()


def retrain(n_folds=10, x, y):
    accuracies = np.zeros((n_folds, ))
    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x), rg)):
        # get the dataset from the correct splits
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        # Train the KNN (we'll use one nearest neighbour)
        knn_classifier = KNNClassifier(k=1)
        knn_classifier.fit(x_train, y_train)
        predictions = knn_classifier.predict(x_test)
        acc = accuracy(y_test, predictions)
        accuracies[i] = acc

        return accuracies.mean()


# print(accuracies)
# print(accuracies.mean())
# we don't need Standard Deviation - print(accuracies.std())
