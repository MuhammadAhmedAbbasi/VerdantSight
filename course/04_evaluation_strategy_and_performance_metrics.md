# Task 4: Evaluation Strategy and Performance Metrics

This task is theory-focused. No code is required.

## Evaluation Strategy

The dataset is divided into three parts:

- **Training set**: used to train the model and update its parameters.
- **Validation set**: used during development to compare model settings and identify the best checkpoint.
- **Test set**: used only after training to estimate how well the final model generalizes to unseen data.

The test set should not be used to tune the model. Otherwise, the final test result would no longer represent an unbiased estimate of performance.

## Accuracy

Accuracy measures the proportion of predictions that are correct.

**Accuracy = Correct Predictions / Total Predictions**

Accuracy is easy to understand and useful when classes are reasonably balanced. However, accuracy alone can hide poor performance on a minority class.

## Precision

Precision answers the question:

> When the model predicts a particular class, how often is that prediction correct?

For a class:

**Precision = True Positives / (True Positives + False Positives)**

High precision means the model produces relatively few false alarms for that class.

## Recall

Recall answers the question:

> Of all real examples belonging to a class, how many did the model correctly identify?

For a class:

**Recall = True Positives / (True Positives + False Negatives)**

High recall means the model misses relatively few real examples of that class.

## F1-Score

The F1-score combines precision and recall into one value.

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

It is useful when both false positives and false negatives matter, especially when the class distribution is uneven.

## Confusion Matrix

A confusion matrix shows how predictions are distributed across the true classes. For VerdantSight, the matrix compares:

- healthy
- chlorosis
- necrosis

It helps answer questions such as:

- Is chlorosis being confused with necrosis?
- Are healthy leaves being incorrectly classified as diseased?
- Is one class significantly harder for the model than the others?

## Per-Class Evaluation

Because VerdantSight contains multiple health categories, metrics should also be examined separately for each class rather than relying only on one overall accuracy value.

For each class, report:

- precision
- recall
- F1-score
- support

**Support** is the number of real test examples belonging to that class.

## Macro and Weighted Averages

**Macro average** calculates the metric for each class and then gives every class equal importance. This is useful when performance on smaller classes matters.

**Weighted average** also calculates the metric per class but weights each class by the number of samples it contains. This reflects the overall dataset distribution more strongly.

## Final Evaluation Approach for VerdantSight

For this project, the final model should be evaluated using:

1. training accuracy to observe learning progress;
2. validation accuracy to compare model versions and select the best checkpoint;
3. final test accuracy on unseen data;
4. per-class precision, recall, and F1-score;
5. a confusion matrix to inspect class-specific errors.

The optimized model in Task 6 uses validation performance to select the best checkpoint and then reports the final metrics on the test set.