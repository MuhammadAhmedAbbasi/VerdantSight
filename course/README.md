# VerdantSight Course Code

This folder separates the VerdantSight leaf-health classification workflow into the exact course tasks.

## Dataset structure

Use a dataset with this structure:

```text
dataset/
├── healthy/
├── chlorosis/
└── necrosis/
```

Supported image formats include JPG, JPEG, PNG, BMP, and WEBP.

## Tasks

1. `01_problem_definition_and_computer_vision_system_design.md` — theory only.
2. `02_data_collection_cleaning_and_preparation.md` — theory only.
3. `03_data_labeling_and_dataset_organization.py` — labels images from folders and creates train/validation/test CSV files.
4. `04_evaluation_strategy_and_performance_metrics.md` — theory only.
5. `05_computer_vision_model_selection_and_baseline_training.py` — trains the baseline ResNet9 model.
6. `06_model_optimization_and_generalization.py` — trains the improved model using augmentation, class balancing, better learning settings, and best-model saving.
7. `07_model_export_and_inference_service_development.py` — exports the optimized model and runs inference on one new image.
8. `08_prediction_output_and_response_structure_design.py` — converts the prediction into a structured JSON response.

Each Python file is self-contained so students can study and run that task independently.

## Installation

```bash
pip install torch torchvision pillow scikit-learn
```

## How to run

Open each Python file and change the path variables near the top when needed.

For example:

```python
DATASET_PATH = Path("dataset")
```

or:

```python
TEST_IMAGE_PATH = Path("test_leaf.jpg")
```

Then run the file normally:

```bash
python course/03_data_labeling_and_dataset_organization.py
python course/05_computer_vision_model_selection_and_baseline_training.py
python course/06_model_optimization_and_generalization.py
python course/07_model_export_and_inference_service_development.py
python course/08_prediction_output_and_response_structure_design.py
```

Task 4 is theory only, so there is no Python file to run for it.

## Baseline and optimized model

Task 5 is intentionally kept simple:

- no data augmentation;
- no class-balanced sampler;
- higher learning rate;
- fewer epochs;
- final model is saved directly.

Task 6 improves the training process using:

- image augmentation;
- `WeightedRandomSampler` for class imbalance;
- learning rate `0.001`;
- more training epochs;
- best validation checkpoint selection.

This gives students a clear sequence from a basic baseline model to a better generalized model.
