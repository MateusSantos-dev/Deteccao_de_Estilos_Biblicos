from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_basic(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], title: str = "Matriz de Confusão") -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.title(title)
    plt.tight_layout()
    plt.show()
