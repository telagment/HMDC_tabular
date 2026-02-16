import numpy as np

def hamming_score_mdc(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.size

def hamming_score_mdc_variate(y_true, y_pred):
    # To deal with the case where the arrays (y_true, y_pred) are of variable length
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return np.sum(y_true == y_pred) / y_true.size

def hamming_loss_mdc_variate(y_true, y_pred):
    # To deal with the case where the arrays (y_true, y_pred) are of variable length
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return 1 - np.sum(y_true == y_pred) / y_true.size

def hamming_loss_mdc(y_true, y_pred):
    return 1 - (np.sum(y_true == y_pred) / y_true.size)


def exact_match_mdc(y_true, y_pred):
    return np.sum(np.all(y_true == y_pred, axis=1)) / len(y_true)


def zero_one_loss_mdc(y_true, y_pred):
    """Handles variable-length arrays by computing row-wise equality."""
    total_instances = len(y_true)
    correct_instances = sum(np.array_equal(y_t, y_p) for y_t, y_p in zip(y_true, y_pred))

    return 1 - (correct_instances / total_instances)

def zero_one_score_mdc(y_true, y_pred):
    """Handles variable-length arrays by computing row-wise equality."""
    total_instances = len(y_true)
    correct_instances = sum(np.array_equal(y_t, y_p) for y_t, y_p in zip(y_true, y_pred))

    return (correct_instances / total_instances)

def sub_exact_match_mdc(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    n_samples, n_targets = y_pred.shape
    corrects = np.sum(y_true == y_pred, axis=1)
    return np.sum(corrects >= n_targets - 1) / n_samples


def per_class_accuracy(y_true, y_pred, y_card, missing_indicator=None):
    """
    Compute per-class accuracy and normalize so that the sum = 1.
    Missing values (marked in missing_indicator) are ignored.
    
    Parameters:
        y_true: np.array of shape (n,) or (1,n) — true labels
        y_pred: np.array of shape (n,) or (1,n) — predicted labels
        y_card: np.array — array of class labels
        missing_indicator: np.array of shape (n,) — 1 if missing, 0 if valid (default: None)
    
    Returns:
        np.array — normalized accuracy per class, sum = 1
    """
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if missing_indicator is None:
        missing_indicator = np.zeros_like(y_true)
    else:
        missing_indicator = missing_indicator.flatten()
    
    # Keep only valid samples
    valid_mask = (missing_indicator == 1)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    acc = np.zeros(len(y_card), dtype=float)
    
    for i, cls in enumerate(y_card):
        y_pred_idx = []
        y_true_idx = []
        idx = (y_true == cls)
        if idx.sum() == 0:
            acc[i] = 0.0  # no valid samples of this class
        else:
            y_pred_idx = y_pred[idx]
            y_true_idx = y_true[idx]
            acc[i] = (y_pred[idx] == y_true[idx]).sum() / idx.sum()

    return acc

def deltas_distance(y_true, y_pred, y_card, missing_indicator=None):
    deltas_max = []
    deltas_ave = []

    for i in range(y_true.shape[1]):
        yi_true = y_true[:, i]
        yi_pred = y_pred[:, i]
        yi_card = np.array(list(y_card.values())[i])
        id_missing = missing_indicator[:, i] if missing_indicator is not None else None
        
        acc = per_class_accuracy(yi_true, yi_pred, yi_card, id_missing)
        # Mean accuracy
        del_mean = acc.mean()
    
        # Difference between max and min
        del_max = np.abs(acc.max() - acc.min())
        deltas_max.append(del_max)
        deltas_ave.append(del_mean)

    return np.array(deltas_max), np.array(deltas_ave)

def deltas_distance(y_true, y_pred, y_card, missing_indicator=None):
    deltas_max = []
    deltas_ave = []

    for i in range(y_true.shape[1]):
        yi_true = y_true[:, i]
        yi_pred = y_pred[:, i]
        yi_card = np.array(list(y_card.values())[i])
        id_missing = missing_indicator[:, i] if missing_indicator is not None else None
        
        acc = per_class_accuracy(yi_true, yi_pred, yi_card, id_missing)
        # Mean accuracy
        del_mean = acc.mean()
    
        # Difference between max and min
        del_max = np.abs(acc.max() - acc.min())
        deltas_max.append(del_max)
        deltas_ave.append(del_mean)
        
    return np.array(deltas_max), np.array(deltas_ave)

