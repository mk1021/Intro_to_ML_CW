import numpy as np
from numpy.random import default_rng

from decision_tree import decision_tree_learning, predict


def confusion_matrix(true_labels, predicted_labels):
    # returns a 4x4 matrix
    # [ TP  FN  |  FP  TN ] ?

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
    # (TP + TN)/(TP + TN + FP + FN) HINT: you can derive the metrics directly
    # from the previously computed confusion matrix

    accuracies = []
    if np.sum(confusion) > 0:
        for i in range(len(confusion)):
            accuracies.append(np.trace(confusion) / np.sum(confusion))
        return accuracies.mean(), accuracies.std()
    else:
        return 0


def precision_rate(confusion):
    # (TP)/(TP + FP)

    precisions = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        col_sum = np.sum(confusion[:, i])
        precisions.append(tp / col_sum)

    macro_p = np.mean(precisions)

    return np.array(precisions), macro_p


def recall_rate(confusion):
    # (TP)/(TP + FN))

    recalls = []
    for i in range(len(confusion)):
        tp = confusion[i, i]
        row_sum = np.sum(confusion[i, :])
        recalls.append(tp / row_sum)

    macro_r = np.mean(recalls)

    return np.array(recalls), macro_r


def f1_score(precisions, recalls):
    # 2(precision)(recall)/(precision + recall)

    # precisions = precision_rate(true_labels, predicted_labels)
    # recalls = recall_rate(true_labels, predicted_labels)

    assert len(precisions) == len(recalls)

    f1 = np.divide(2 * np.multiply(precisions, recalls), precisions + recalls)

    macro_f = np.mean(f1)

    return f1, macro_f


def evaluate(true_labels, predicted_labels):
    confusion = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy(confusion)
    class_precisions, macro_precision = precision_rate(confusion)
    class_recalls, macro_recall = recall_rate(confusion)
    f1_scores, macro_f1 = f1_score(class_precisions, class_recalls)

    print(f"Confusion Matrix: \n {confusion}")
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


# data_clean = np.loadtxt('data/noisy_dataset.txt')
#
# test_total = round(len(data_clean) * 0.2)
# rg = default_rng(60012)
# rand_indexes = rg.permutation(len(data_clean))
# test = data_clean[rand_indexes[:test_total]]
# train = data_clean[rand_indexes[test_total:]]
#
# decision_tree = decision_tree_learning(train, 0)[0]
# max_depth = decision_tree_learning(train, 0)[1]
# y_prediction = predict(decision_tree, test[:, :-1])
# y_gold = test[:, -1]
#
# print(evaluate(y_gold, y_prediction))
