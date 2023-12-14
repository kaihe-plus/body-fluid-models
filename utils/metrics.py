from typing import List

import numpy as np
from imblearn.metrics import specificity_score
from numpy import ndarray
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc

metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 score', 'MCC', 'AP', 'PRAUC']


def pr_auc_score(y_true: ndarray, y_score: ndarray):
    pr, re, _ = precision_recall_curve(y_true, y_score)
    return auc(re, pr)


def eval_print(y_true: ndarray, y_pred: ndarray, y_score: ndarray, label_str=None):
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    re = recall_score(y_true, y_pred)
    sp = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ap = average_precision_score(y_true, y_score)
    pr_auc = pr_auc_score(y_true, y_score)
    if label_str is not None:
        print('{:s} performance: Acc({:.6f}), PR({:.6f}), RE/SE({:.6f}), SP({:.6f}), F1({:.6f}), MCC({:.6f}), '
              'AP({:.6f}), PRAUC({:.6f})'.format(label_str, acc, pr, re, sp, f1, mcc, ap, pr_auc))
    return acc, pr, re, sp, f1, mcc, ap, pr_auc


def select_threshold_by_mcc(y_true: ndarray, y_score: ndarray):
    pred_values = np.unique(
        np.round(y_score, 2)
    )
    np.sort(pred_values)

    best_scores = -1.
    best_threshold = pred_values[0]
    for pred_value in pred_values:
        y_pred = np.asarray(y_score > pred_value, dtype=int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_scores:
            best_scores = mcc
            best_threshold = pred_value
    return best_threshold


def eval_mean_print(
        y_true_list: List[ndarray],
        y_pred_list: List[ndarray],
        y_score_list: List[ndarray],
        label_str=None
):
    metrics_list = []
    for y_true, y_pred, y_score in zip(y_true_list, y_pred_list, y_score_list):
        metrics = eval_print(y_true, y_pred, y_score)
        metrics_list.append(metrics)
    metrics_array = np.array(metrics_list, dtype=float)
    mean_metrics = np.mean(metrics_array, axis=0)
    if label_str is not None:
        print('{:s} performance: Acc({:.6f}), PR({:.6f}), RE/SE({:.6f}), SP({:.6f}), F1({:.6f}), MCC({:.6f}), '
              'AP({:.6f}), PRAUC({:.6f})'.format(label_str, *mean_metrics))
    return metrics_array, mean_metrics.tolist()


if __name__ == '__main__':
    my_y_true = [
        np.asarray(
            np.random.uniform(0, 1, [100]) > 0.9,
            dtype=int
        )
        for i in range(10)
    ]
    my_y_pred = [
        np.asarray(
            np.random.uniform(0, 1, [100]) > 0.9,
            dtype=int
        )
        for i in range(10)
    ]
    my_y_score = [
        np.random.uniform(0, 1, [100])
        for i in range(10)
    ]

    for i in range(10):
        eval_print(my_y_true[i], my_y_pred[i], my_y_score[i], 'Task {:02d}'.format(i))

    eval_mean_print(my_y_true, my_y_pred, my_y_score, 'Mean')
