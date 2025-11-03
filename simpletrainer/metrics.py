import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score, accuracy_score
)

def binary_metrics(labels, outputs, use_optimal_threshold=True):
    labels = np.array(labels)
    outputs = np.array(outputs)
    
    # handle both single logit and 2-class output
    if outputs.ndim > 1 and outputs.shape[1] == 2:
        probs = outputs[:, 1]
    else:
        probs = outputs.ravel()
        probs = 1 / (1 + np.exp(-probs))  # sigmoid

    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5

    if use_optimal_threshold and len(set(labels)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        threshold = 0.5

    preds = (probs >= threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (preds == labels).mean()

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }

def multiclass_metrics(labels, outputs):
    labels = np.array(labels)
    outputs = np.array(outputs)
    preds = outputs.argmax(axis=1)
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

def regression_metrics(labels, outputs):
    labels = np.array(labels)
    outputs = np.array(outputs)
    mse = mean_squared_error(labels, outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, outputs)
    return {"mse": mse, "rmse": rmse, "r2": r2, "loss": mse}

def multilabel_metrics(labels, outputs, threshold=0.5):
    probs = 1 / (1 + np.exp(-outputs)) if outputs.max() > 1 else outputs
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }
    return metrics
