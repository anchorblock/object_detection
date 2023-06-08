import evaluate
import datasets
import typing
import numpy as np
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score, average_precision_score


def top_k_function(predictions, references):
    """Returns the accuracy at k"""
    if isinstance(predictions, list):
        accuracyk = sum(
            [reference in kpredictions for kpredictions, reference in zip(predictions, references)]
        ) / len(references)    
    else:
        accuracyk = (
            references[:, None] == predictions[:, :]
        ).any(axis=1).sum() / len(references)
    return accuracyk


def compute_metrics_imagenet1k(predictions, references):

    predictions = predictions.numpy()
    references = references.numpy()


    labels = list(range(1000))
    counts_each_label = [np.count_nonzero(references == label) for label in labels]


    # Top-1 accuracy
    k = 1
    topk_indices = np.argsort(-predictions, axis=1)[:, :k] # descending order
    top1_accuracy_raw = top_k_function(references = references, predictions = topk_indices)*len(predictions)

    # Top-5 Accuracy
    k = 5
    topk_indices = np.argsort(-predictions, axis=1)[:, :k] # descending order
    top5_accuracy_raw = top_k_function(references = references, predictions = topk_indices)*len(predictions)

    # Precision

    precision_result = precision_score(
                    y_pred = predictions.argmax(axis=1), 
                    y_true = references, 
                    labels=labels, 
                    pos_label=1, 
                    average=None, 
                    zero_division=0)
    
    # Recall

    recall_result = recall_score(
                    y_pred = predictions.argmax(axis=1), 
                    y_true = references, 
                    labels=labels, 
                    pos_label=1, 
                    average=None, 
                    zero_division=0)
    

    # F1-Score

    f1_result = f1_score(
                    y_pred = predictions.argmax(axis=1), 
                    y_true = references, 
                    labels=labels, 
                    pos_label=1, 
                    average=None, 
                    zero_division=0)
    

    # Mean Average Precision (mAP)
    mAP_result = []
    for i in range(1000):
        mAP_result.append(average_precision_score(references == i, predictions[:, i]))



    # json initialize

    results = {
        "topk_accuracy": {
            "top1_accuracy_raw" : top1_accuracy_raw,
            "top5_accuracy_raw" : top5_accuracy_raw,
            "n_samples": len(predictions)
        },
        "precision_recall_f1_mAP": {}
        }
    

    for i in range(1000):

        n_samples_of_label_i = counts_each_label[i]

        labelwise_results = {
            "precision_raw": precision_result[i]*n_samples_of_label_i,
            "recall_raw": recall_result[i]*n_samples_of_label_i,
            "f1_raw": f1_result[i]*n_samples_of_label_i,
            "mAP_raw": mAP_result[i]*n_samples_of_label_i,
            "n_samples": n_samples_of_label_i
        }
        results["precision_recall_f1_mAP"][str(i)] = labelwise_results


    return results

