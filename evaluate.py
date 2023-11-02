import numpy as np


def confusion_matrix(true_labels, predicted_labels):
    # returns a 4x4 matrix
    # [ TP  FN  |  FP  TN ] ?

    # Define class labels
    class_labels = np.unique(np.concatenate((true_labels, predicted_labels)),
                             dtype=np.int)

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
    # (TP + TN)/(TP + TN + FP + FN) HINT: you can derive the metrics directly
    # from the previously computed confusion matrix

    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0


def precision_rate(confusion):
    # (TP)/(TP + FP)

    precision = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        col_sum = np.sum(confusion[:, i])
        precision.append(tp / col_sum)

    return np.array(precision)


def recall_rate(confusion):
    # (TP)/(TP + FN))

    recall = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        row_sum = np.sum(confusion[i, :])
        recall.append(tp / row_sum)

    return np.array(recall)


def f1_score(precisions, recalls):
    # 2(precision)(recall)/(precision + recall)

    # precisions = precision_rate(true_labels, predicted_labels)
    # recalls = recall_rate(true_labels, predicted_labels)

    assert len(precisions) == len(recalls)

    f1 = np.divide(2 * np.multiply(precisions, recalls), precisions + recalls)

    return f1
