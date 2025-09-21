from sklearn.svm import SVC


def create_svm(**params) -> SVC:
    default_params = {
        'C': 1.0,
        'kernel': 'linear',
        'class_weight': 'balanced',
        'random_state': 99,
        'probability': True
    }
    model_params = {**default_params, **params}
    return SVC(**model_params)


def print_svm_info(model: SVC):
    print("SVM INFO:")
    print(f"Kernel: {model.kernel}")
    print(f"C: {model.C}")
    print(f"Number of support vectors: {len(model.support_vectors_)}")
