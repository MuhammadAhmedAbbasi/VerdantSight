# Task 1: Problem Definition and Computer Vision System Design

## Problem Definition

VerdantSight is designed to identify the health condition of pothos leaves from images. The classification model predicts one of three classes:

- `healthy`
- `chlorosis`
- `necrosis`

The input to the classifier is an RGB image of a leaf. The output is a predicted health class and an associated confidence score.

## Computer Vision Task

This course focuses on **image classification**. The wider VerdantSight application can first detect a leaf region and then pass that crop to the classifier, but the model developed in this course is the health classifier itself.

## System Design

```text
Camera / image source
        |
        v
Leaf image or detected leaf crop
        |
        v
Image preprocessing
        |
        v
ResNet9 classifier
        |
        v
Healthy / Chlorosis / Necrosis
        |
        v
Confidence score and structured response
```

## Design Requirements

The classifier should:

1. accept images captured under realistic conditions;
2. distinguish three visually different health states;
3. generalize to images not seen during training;
4. provide confidence values together with predictions;
5. be lightweight enough to support a real-time inference pipeline.

## Main Challenges

Important challenges include variation in lighting, orientation, image background, leaf size, camera quality, and unequal numbers of examples across the three classes.
