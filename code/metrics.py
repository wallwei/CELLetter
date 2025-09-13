import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve
import torch
import torch.nn.functional as F

def calculate_metrics(all_labels, all_preds, all_probs, criterion=None, test_loader=None):
    """Calculate evaluation metrics"""
    # Calculate ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': roc_thresholds
    })

    # Calculate PR curve data
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    pr_data = pd.DataFrame({
        'Recall': recall,
        'Precision': precision,
        'Threshold': np.append(pr_thresholds, 1)  # Add the last point
    })

    # Calculate MCC
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics = {
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "aupr": average_precision_score(all_labels, all_probs),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "mcc": mcc,
        "roc_data": roc_data,
        "pr_data": pr_data
    }

    if criterion is not None and test_loader is not None:
        metrics["loss"] = calculate_loss(criterion, all_labels, all_preds, test_loader)

    return metrics, roc_data, pr_data

def calculate_loss(criterion, all_labels, all_preds, test_loader):
    """Calculate loss if needed"""
    # This is a placeholder - you might need to adjust based on your actual loss calculation
    return 0.0