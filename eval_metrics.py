from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score, precision_recall_curve, auc
import numpy as np


def perform_metrics(pred, gt, mask):
    # Suppressing background regions.
    pred = pred[mask > 0]
    gt = gt[mask > 0]

    # prob = prob[mask > 0]

    threshold_confusion = 0.5
    y_pred = np.empty((pred.shape[0]))
    for i in range(pred.shape[0]):
        if pred[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    # Building confusion matrix.
    c_matrix = confusion_matrix(gt, y_pred)

    # Calculating ratios.
    tn = c_matrix[0, 0]
    tp = c_matrix[1, 1]
    fn = c_matrix[1, 0]
    fp = c_matrix[0, 1]

    # Finding the metrics.
    precision = tp / (tp + fp)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(gt, y_pred, labels=None, average='binary', sample_weight=None)
    roc_auc = roc_auc_score(gt, pred)

    pre, rec, thres = precision_recall_curve(gt, pred)
    pr_auc = auc(rec, pre)

    return precision, sen, spec, f1, acc, roc_auc, pr_auc

