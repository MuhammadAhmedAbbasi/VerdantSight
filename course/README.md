# VerdantSight Course Code

This folder separates the VerdantSight leaf-health classification workflow into the exact course tasks.

## Dataset structure

Place or point the scripts to a dataset with this structure:

```text
dataset/
├── healthy/
├── chlorosis/
└── necrosis/
```

The images can be JPG, JPEG, PNG, BMP, or WEBP.

## Tasks

1. `01_problem_definition_and_computer_vision_system_design.md` — conceptual task, no code.
2. `02_data_collection_cleaning_and_preparation.md` — documents the real offline/manual collection and cleaning process, no code.
3. `03_data_labeling_and_dataset_organization.py` — assigns labels from folders and creates reproducible train/validation/test manifests.
4. `04_evaluation_strategy_and_performance_metrics.py` — demonstrates accuracy, precision, recall, F1-score, and confusion matrix.
5. `05_computer_vision_model_selection_and_baseline_training.py` — trains an intentionally simple/weak ResNet9 baseline.
6. `06_model_optimization_and_generalization.py` — trains the improved model using augmentation, class-balanced sampling, better learning settings, and best-model checkpointing.
7. `07_model_export_and_inference_service_development.py` — loads the optimized model, exports a deployment checkpoint, and provides a reusable inference service class.
8. `08_prediction_output_and_response_structure_design.py` — converts inference results into a structured JSON response.

`common.py` contains shared architecture and utility code so every task uses exactly the same class order and ResNet9 definition.

## Installation

```bash
pip install torch torchvision pillow scikit-learn
```

## Recommended run order

From the repository root:

```bash
python course/03_data_labeling_and_dataset_organization.py --dataset path/to/dataset
python course/04_evaluation_strategy_and_performance_metrics.py
python course/05_computer_vision_model_selection_and_baseline_training.py --dataset path/to/dataset
python course/06_model_optimization_and_generalization.py --dataset path/to/dataset
python course/07_model_export_and_inference_service_development.py --image path/to/test_leaf.jpg
python course/08_prediction_output_and_response_structure_design.py --image path/to/test_leaf.jpg
```

All generated course results are written under `course/artifacts/`.

## Baseline versus optimized model

The Task 5 baseline is intentionally limited for teaching purposes:

- no data augmentation;
- ordinary shuffled batches, without class balancing;
- deliberately suboptimal learning rate;
- fewer epochs;
- final-epoch model is saved rather than choosing the best validation checkpoint.

Task 6 then introduces the stronger settings used by the project:

- geometric and color augmentation;
- `WeightedRandomSampler` for class imbalance;
- Adam with a more appropriate learning rate;
- more training epochs;
- best validation checkpoint selection.

This creates a clear experimental story: establish a weak but valid baseline, measure it, then improve generalization systematically.
