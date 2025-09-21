from sklearn.model_selection import StratifiedKFold
from src.evaluation.metrics import evaluate_basic
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import numpy as np


def evaluate_cross_validation(model, x, y, num_folds: int = 5, random_state: int = 99) -> dict:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    splits = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    for train_idx, val_idx in splits.split(x, y_encoded):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train_encoded, y_val_encoded = y_encoded[train_idx], y_encoded[val_idx]

        fold_model = clone(model)
        fold_model.fit(x_train, y_train_encoded)
        y_pred_encoded = fold_model.predict(x_val)

        metrics = evaluate_basic(y_val_encoded, y_pred_encoded)
        fold_metrics.append(metrics)
        all_y_true.extend(y_val_encoded)
        all_y_pred.extend(y_pred_encoded)

    accuracy_scores = [fold_metric["accuracy"] for fold_metric in fold_metrics]
    f1_scores = [fold_metric["f1"] for fold_metric in fold_metrics]

    results = {
        "mean_accuracy": np.mean(accuracy_scores),
        "std_accuracy": np.std(accuracy_scores),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
        "fold_metrics": fold_metrics,
        "confusion_matrix_data": (all_y_true, all_y_pred)
    }

    return results
