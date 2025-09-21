from sklearn.naive_bayes import MultinomialNB


def create_naive_bayes(**params) -> MultinomialNB:
    default_params = {
        "alpha": 1.0,
        "fit_prior": True,
        'class_prior': None
    }
    model_params = {**default_params, **params}
    return MultinomialNB(**model_params)


def print_naive_bayes_info(model: MultinomialNB, feature_names: list = None):
    print("NAIVE BAYES INFO:")
    print(f"Alpha (suavização): {model.alpha}")
    print(f"Classes: {model.classes_}")
    print(f"Log prior probabilities: {model.class_log_prior_}")

    if feature_names and hasattr(model, 'feature_log_prob_'):
        print(f"Number of features: {model.feature_log_prob_.shape[1]}")
        for i, class_name in enumerate(model.classes_):
            print(f"Top features for class '{class_name}':")
            top_indices = model.feature_log_prob_[i].argsort()[-5:][::-1]
            for idx in top_indices:
                feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"      {feature_name}: {model.feature_log_prob_[i][idx]:.4f}")