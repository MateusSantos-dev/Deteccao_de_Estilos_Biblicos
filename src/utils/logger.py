def print_cross_validation_results(results: dict, model_name: str = "Modelo") -> None:
    print(f"\n{'=' * 50}")
    print(f"results cross validation model: {model_name.upper()}")
    print(f"{'=' * 50}")

    print(f"mean accuracy: {results['mean_accuracy']:.4f} (±{results['std_accuracy']:.4f})")
    print(f"mean f1: {results['mean_f1']:.4f} (±{results['std_f1']:.4f})")

    print(f"\nper fold")
    for i, fold_metrics in enumerate(results['fold_metrics'], 1):
        print(f"   Fold {i}: accuracy = {fold_metrics['accuracy']:.4f}, f1 = {fold_metrics['f1']:.4f}")