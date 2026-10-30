import numpy as np


def calculate_confusion_matrix(y_true, y_pred_prob, threshold=0.5):
    """
    Computes confusion matrix entries: TP, FP, TN, FN.
    """
    y_true = np.array(y_true).ravel()
    y_pred = (np.array(y_pred_prob).ravel() >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    cm = calculate_confusion_matrix(y_true, y_pred_prob, threshold)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr, tpr, roc_auc = compute_roc_curve(y_true, y_pred_prob)

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "Confusion_Matrix": cm,
        "ROC_FPR": fpr,
        "ROC_TPR": tpr
    }


def compute_roc_curve(y_true, y_pred_prob):
    """
    Computes FPR, TPR arrays and trapezoidal ROC-AUC score.
    """
    y_true = np.array(y_true).ravel()
    y_pred_prob = np.array(y_pred_prob).ravel()

    # Sort thresholds in descending order
    desc_indices = np.argsort(y_pred_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    y_prob_sorted = y_pred_prob[desc_indices]

    distinct_thresholds = np.unique(y_prob_sorted)[::-1]
    
    fpr_list = [0.0]
    tpr_list = [0.0]

    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)

    if num_positives == 0 or num_negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    for thresh in distinct_thresholds:
        preds = (y_prob_sorted >= thresh).astype(int)
        tp = np.sum((y_true_sorted == 1) & (preds == 1))
        fp = np.sum((y_true_sorted == 0) & (preds == 1))

        tpr_list.append(tp / num_positives)
        fpr_list.append(fp / num_negatives)

    fpr_list.append(1.0)
    tpr_list.append(1.0)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Sort FPR to ensure monotonically non-decreasing for trapezoidal integration
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    tpr_arr = tpr_arr[sort_idx]

    # Trapezoidal rule integration for AUC
    roc_auc = np.trapz(tpr_arr, fpr_arr)
    roc_auc = float(np.clip(roc_auc, 0.0, 1.0))

    return fpr_arr, tpr_arr, roc_auc
