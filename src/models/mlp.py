from sklearn.neural_network import MLPClassifier

def create_mlp(**params) -> MLPClassifier:
    default_params = {
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 1000,
        'random_state': 99,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 15,
        'alpha': 0.0001,
        'learning_rate_init': 0.001,
    }
    model_params = {**default_params, **params}
    return MLPClassifier(**model_params)


def print_mlp_info(model: MLPClassifier):
    print("MLP INFO:")
    print(f"Architecture: {model.hidden_layer_sizes}")
    print(f"Activation: {model.activation}")
    print(f"Solver: {model.solver}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Final loss: {model.loss_:.4f}")

    if hasattr(model, 'validation_scores_'):
        print(f"Best validation score: {max(model.validation_scores_):.4f}")