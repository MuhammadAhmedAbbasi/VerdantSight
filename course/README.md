# VerdantSight Course Code

This folder separates the VerdantSight leaf-health classification workflow into the exact course tasks.

Each coded task is self-contained. Students can open and run the relevant task file without importing code from another course file.

## Dataset structure

Use a dataset with this structure:

```text
dataset/
├── healthy/
├── chlorosis/
└── necrosis/
```

Supported image formats are JPG, JPEG, PNG, BMP, and WEBP.

## Tasks

1. `01_problem_definition_and_computer_vision_system_design.md` — theory only.
2. `02_data_collection_cleaning_and_preparation.md` — theory only; documents offline/manual collection and cleaning.
3. `03_data_labeling_and_dataset_organization.py` — assigns labels from folders and creates reproducible train/validation/test split files.
4. `04_evaluation_strategy_and_performance_metrics.md` — theory only; explains accuracy, precision, recall, F1-score, confusion matrix, and evaluation strategy.
5. `05_computer_vision_model_selection_and_baseline_training.py` — self-contained baseline ResNet9 training experiment.
6. `06_model_optimization_and_generalization.py` — self-contained optimized ResNet9 training experiment.
7. `07_model_export_and_inference_service_development.py` — self-contained model export and inference example.
8. `08_prediction_output_and_response_structure_design.py` — self-contained prediction-to-JSON response example.

## Installation

```bash
pip install -r course/requirements.txt
```

## Run order

Task 3:

```bash
python course/03_data_labeling_and_dataset_organization.py --dataset path/to/dataset
```

Task 4 is theory only and does not require execution.

Task 5:

```bash
python course/05_computer_vision_model_selection_and_baseline_training.py --dataset path/to/dataset
```

Task 6:

```bash
python course/06_model_optimization_and_generalization.py --dataset path/to/dataset
```

Task 7:

```bash
python course/07_model_export_and_inference_service_development.py --image path/to/test_leaf.jpg
```

Task 8:

```bash
python course/08_prediction_output_and_response_structure_design.py --image path/to/test_leaf.jpg
```

Generated models and outputs are stored under `course/artifacts/`.

## Baseline versus optimized model

Task 5 intentionally uses a weaker but valid training configuration:

- no data augmentation;
- no class-balanced sampling;
- only 5 epochs;
- Adam learning rate of `0.01`;
- final epoch is saved instead of selecting the best validation checkpoint.

Task 6 introduces the improved configuration:

- random horizontal flipping;
- random rotation;
- color jitter;
- `WeightedRandomSampler` for class imbalance;
- Adam learning rate of `0.001`;
- 20 epochs;
- best validation checkpoint selection.

This gives the course a clear experimental sequence: build a baseline, observe its limitations, improve the training strategy, then export and use the optimized model.