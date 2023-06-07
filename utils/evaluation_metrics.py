### INCOMPLETE SCRIPT
### WILL PUBLISH IN NEXT RELEASE


### following evaluation metrics are popular for imagenet-1k:

            # Top-1 Accuracy
            # Top-5 Accuracy
            # Precision
            # Recall
            # F1-Score
            # Mean Average Precision (mAP)


import evaluate

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)






import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, num_classes):
    # Top-1 Accuracy
    top1_accuracy = np.mean(np.argmax(y_pred, axis=1) == y_true)

    # Top-5 Accuracy
    top5_accuracy = np.mean(np.any(np.argsort(y_pred, axis=1)[:, -5:] == np.expand_dims(y_true, axis=1), axis=1))

    # Precision
    precision = precision_score(y_true, np.argmax(y_pred, axis=1), average='macro')

    # Recall
    recall = recall_score(y_true, np.argmax(y_pred, axis=1), average='macro')

    # F1-Score
    f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')

    # Confusion Matrix
    confusion_mat = confusion_matrix(y_true, np.argmax(y_pred, axis=1))

    # Mean Average Precision (mAP)
    y_true_one_hot = np.eye(num_classes)[y_true]
    mAP = np.mean(np.average(y_true_one_hot[np.argsort(-y_pred)], axis=1))

    return top1_accuracy, top5_accuracy, precision, recall, f1, confusion_mat, mAP



import torch

def compute_top5_accuracy(predictions, labels):
    _, top5_predictions = torch.topk(predictions, k=5, dim=1)
    correct = top5_predictions.eq(labels.view(-1, 1).expand_as(top5_predictions))
    top5_accuracy = correct.sum().item() / labels.size(0)
    return top5_accuracy

########################################33

import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")
top_5_accuracy = evaluate.load("top_5_accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1_score = evaluate.load("f1_score")
mean_average_precision = evaluate.load("mean_average_precision")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    top_1_acc = accuracy.compute(predictions=predictions, references=labels)
    top_5_acc = top_5_accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels)
    rec = recall.compute(predictions=predictions, references=labels)
    f1 = f1_score.compute(predictions=predictions, references=labels)
    mAP = mean_average_precision.compute(predictions=predictions, references=labels)

    return {
        "Top-1 Accuracy": top_1_acc,
        "Top-5 Accuracy": top_5_acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Mean Average Precision (mAP)": mAP
    }




