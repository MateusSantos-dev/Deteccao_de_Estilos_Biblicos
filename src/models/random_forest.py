from sklearn.ensemble import RandomForestClassifier
import numpy as np


def create_random_forest(**params) -> RandomForestClassifier:
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 99,
        'n_jobs': -1
    }
    model_params = {**default_params, **params}
    return RandomForestClassifier(**model_params)


def print_random_forest_info(model: RandomForestClassifier, feature_names: list = None, top_n: int = 10):
    print("RANDOM FOREST INFO:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"OOB score: {model.oob_score_ if hasattr(model, 'oob_score_') else 'Not calculated'}")

    if hasattr(model, "feature_importances_") and feature_names:
        print(f"Top {top_n} most important features:")

        feature_importance = list(zip(feature_names, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for i, (name, importance) in enumerate(feature_importance[:top_n]):
            print(f"      {i + 1:2d}. {name}: {importance:.4f}")


def get_top_features(model: RandomForestClassifier, feature_names: list, top_n: int = 10) -> list:
    if hasattr(model, "feature_importances_") and feature_names:
        indices = np.argsort(model.feature_importances_)[::-1][:top_n]
        return [(feature_names[i], model.feature_importances_[i]) for i in indices]
    return []
