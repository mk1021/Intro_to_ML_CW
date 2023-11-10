import numpy as np


def confusion_matrix(true_labels, predicted_labels):
    """ Compute the confusion matrix

        Args:
            true_labels (np.ndarray): the correct ground truth/gold standard labels
            predicted_labels (np.ndarray): the predicted labels

        Returns:
            confusion (np.array) : shape (C, C), where C is the number of classes.
                       Rows are ground truth per class, columns are predictions
        """

    # Define class labels
    class_labels = np.unique(np.concatenate((true_labels, predicted_labels)))

    # Initialize the confusion matrix
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # Iterate through each class label
    for i, label in enumerate(class_labels):
        # Get the predicted labels that match the true label
        true_label_predictions = predicted_labels[true_labels == label]

        # Calculate the counts of unique predicted labels
        unique_predicted_labels, counts = np.unique(true_label_predictions,
                                                    return_counts=True)

        # Create a dictionary to store predicted label frequencies
        frequency_dict = dict(zip(unique_predicted_labels, counts))

        # Fill the confusion matrix based on the current true label
        for j, class_label in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def accuracy(confusion):
    """ Compute the accuracy given the ground truth and predictions

        Args:
            confusion (np.array) = shape (C, C), where C is the number of
            classes. Rows are ground truth per class, columns are predictions

        Returns:
            float : the accuracy
        """

    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0


def precision_rate(confusion):
    """ Compute the precision score per class given the ground truth and
        predictions

        Also return the macro-averaged precision across classes.

        Args:
            confusion (np.array) = shape (C, C), where C is the number of
            classes. Rows are ground truth per class, columns are predictions

        Returns:
            tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.array of shape (C,), where each element is
              the precision for class c
            - macro-precision is macro-averaged precision (a float)
    """

    precisions = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        col_sum = np.sum(confusion[:, i])
        precisions.append(tp / col_sum)

    macro_p = np.mean(precisions)

    return np.array(precisions), macro_p


def recall_rate(confusion):
    """ Compute the recall score per class given the ground truth and
        predictions

        Also return the macro-averaged recall across classes.

        Args:
            confusion (np.array) = shape (C, C), where C is the number of
            classes. Rows are ground truth per class, columns are predictions

        Returns:
            tuple: returns a tuple (recalls, macro_recall) where
                - recalls is a np.array of shape (C,), where each element is
                  the recall for class c
                - macro-recall is macro-averaged recall (a float)
    """

    recalls = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        row_sum = np.sum(confusion[i, :])
        recalls.append(tp / row_sum)

    macro_r = np.mean(recalls)

    return np.array(recalls), macro_r


def f1_score(precisions, recalls):
    """ Compute the F1-score per class given the ground truth and predictions

        Also return the macro-averaged F1-score across classes.

        Args:
            precisions (np.array): precision score per class
            recalls (np.array): recall score per class

        Returns:
            tuple: returns a tuple (f1s, macro_f1) where
                - f1s is a np.ndarray of shape (C,), where each element is the
                  f1-score for class c
                - macro-f1 is macro-averaged f1-score (a float)
        """

    assert len(precisions) == len(recalls)

    f1 = np.divide(2 * np.multiply(precisions, recalls), precisions + recalls)

    macro_f = np.mean(f1)

    return f1, macro_f


def evaluate(average_confusion):
    acc = accuracy(average_confusion)
    class_precisions, macro_precision = precision_rate(average_confusion)
    class_recalls, macro_recall = recall_rate(average_confusion)
    f1_scores, macro_f1 = f1_score(class_precisions, class_recalls)

    print(f"Confusion Matrix: \n {average_confusion}")
    print()
    print(f"Accuracy: {acc}")
    print()
    print("Precision: ")
    for i in range(len(class_precisions)):
        print(f"\tClass {i}: {class_precisions[i]}")
    print(f"Macro-averaged Precision: {macro_precision}")
    print()
    print("Recall: ")
    for i in range(len(class_recalls)):
        print(f"\tClass {i}: {class_recalls[i]}")
    print(f"Macro-averaged Recall: {macro_recall}")
    print()
    print("F1 Scores: ")
    for i in range(len(f1_scores)):
        print(f"\tClass {i}: {f1_scores[i]}")
    print(f"Macro-averaged F1 Score: {macro_f1}")

