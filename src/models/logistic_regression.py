from sklearn.linear_model import LogisticRegression


def create_logistic_regression(**params) -> LogisticRegression:
    default_params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 99,
        "n_jobs": 1
    }
    model_params = {**default_params, **params}
    return LogisticRegression(**model_params)


def print_logistic_regression_info(model: LogisticRegression, feature_names: list = None, top_n: int = 10):
    print("LOGISTIC REGRESSION INFO:")
    print(f"Iterations: {model.n_iter_}")
    print(f"Classes: {model.classes_}")

    if hasattr(model, "coef_") and feature_names:
        print(f"Top {top_n} most important features:")
        for i, class_coef in enumerate(model.coef_):
            top_indices = class_coef.argsort()[-top_n:][::-1]
            for j, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"Class {i} - {j + 1:2d}. {feature_name}: {class_coef[idx]:.4f}")