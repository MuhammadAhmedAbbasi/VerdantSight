# Task 2: Data Collection, Cleaning, and Preparation

## Data Collection

The images used for VerdantSight were collected offline. The dataset combines manually collected leaf photographs with image data downloaded previously from external sources. No automated web-scraping or online collection code is used in this task.

## Cleaning Process

Before model training, the collected images are reviewed manually. The cleaning process includes:

- removing corrupted or unreadable files;
- removing images in which the target leaf is not sufficiently visible;
- removing very poor-quality or unusable images;
- removing obvious duplicate images where appropriate;
- checking that each image belongs to the correct health category;
- retaining supported image formats such as JPG, JPEG, PNG, BMP, and WEBP.

## Preparation

After cleaning, the images are placed into one folder for each health class:

```text
dataset/
├── healthy/
├── chlorosis/
└── necrosis/
```

This folder structure is important because Task 3 uses the folder name as the image-level class label.

## Output of This Task

The output is a cleaned offline dataset organized into the three class folders and ready for labeling, splitting, and model development.
