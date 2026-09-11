"""Task 4: Evaluation Strategy and Performance Metrics.

This file demonstrates the main classification metrics before model training.
It uses a small example prediction set so it can run independently.

Usage:
    python course/04_evaluation_strategy_and_performance_metrics.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from common import CLASS_NAMES


# Example ground-truth labels and model predictions.
# Class order: 0=healthy, 1=chlorosis, 2=necrosis
Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
Y_PRED = [0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 2, 1]


def main():
    accuracy = accuracy_score(Y_TRUE, Y_PRED)

    precision, recall, f1, support = precision_recall_fscore_support(
        Y_TRUE,
        Y_PRED,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    print("Evaluation metrics")
    print("==================")
    print(f"Overall accuracy: {accuracy:.4f}\n")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"Class: {class_name}")
        print(f"  Precision: {precision[index]:.4f}")
        print(f"  Recall:    {recall[index]:.4f}")
        print(f"  F1-score:  {f1[index]:.4f}")
        print(f"  Support:   {support[index]}")
        print()

    print("Classification report")
    print("---------------------")
    print(
        classification_report(
            Y_TRUE,
            Y_PRED,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("----------------")
    print(confusion_matrix(Y_TRUE, Y_PRED))

    print("\nInterpretation")
    print("--------------")
    print("Accuracy summarizes total correctness.")
    print("Precision measures how reliable each predicted class is.")
    print("Recall measures how many real examples of each class are detected.")
    print("F1-score balances precision and recall.")
    print("The confusion matrix shows which classes are confused with one another.")


if __name__ == "__main__":
    main()
