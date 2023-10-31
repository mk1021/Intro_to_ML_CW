import numpy as np


def confusion_matrix(true_labels, predicted_labels):
    # HINT: you should get a single 4x4 matrix

    # [ TP  FN  |  FP  TN ] ?

    class_labels = np.unique(np.concatenate((true_labels, predicted_labels)),
                             dtype=np.int)
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
    # (TP + TN)/(TP + TN + FP + FN) HINT: you can derive the metrics directly
    # from the previously computed confusion matrix

    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0


def precision_rate(true_labels, predicted_labels):
    # (TP)/(TP + FP)

    unique_class_labels = np.unique(
        np.concatenate(true_labels, predicted_labels))
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

    unique_class_labels = np.unique(
        np.concatenate(true_labels, predicted_labels))
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
            f_one = (2 * current_p * current_r) / (current_p + current_r)
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
