# Driver Drowsiness Detection

## Overview

This repository contains code and models for a machine learning (ML) project focused on developing a drowsiness detection system. The system utilizes various image classification models to predict whether a person's eyes are open or closed, indicating alertness or drowsiness. 

## Key Features

- **Multiple Model Architectures:** Explores the use of Convolutional Neural Networks (CNNs), Multi-Layer Perceptrons (MLPs), and Support Vector Machines (SVMs) for drowsiness detection.
- **Transfer Learning:** Leverages pre-trained CNN models (MobileNetV2) and fine-tuning for improved performance, especially with limited real-world data.
- **Synthetic Data:** Incorporates synthetic eye images to augment the training process and enhance model generalization.
- **Evaluation Framework:** Provides scripts for model evaluation using metrics like accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

## Repository Structure

├── data
│   ├── raw       (Original, unprocessed eye image datasets)
│   ├── processed  (Cleaned, preprocessed images ready for training)
│   └── synthetic  (Synthetically generated eye images)
├── models
│   ├── cnn       (Convolutional Neural Networks - MobileNetV2)
│   │   ├── real_eyes (Models trained on real eye data)
│   │   │   ├── last_layers (Fine-tuning only the last few layers)
│   │   │   └── full_network (Fine-tuning the entire network)
│   │   ├── synthetic_eyes (Models trained on synthetic eye data)
│   │   │   ├── last_layers
│   │   │   └── full_network
│   │   └── fine_tuned_real_eyes (Models fine-tuned from synthetic to real)
│   │       ├── last_layers
│   │       └── full_network
│   ├── mlp       (Multi-Layer Perceptron models)
│   │   ├── ...    (Structure similar to CNN models)
│   └── svm       (Support Vector Machine models)
│       ├── ...
├── src
│   ├── init.py
│   ├── utils.py           (Data loading, preprocessing, etc.)
│   └── model_evaluation.py (Evaluation scripts)
├── .gitignore
├── README.md  (This file)
├── requirements.txt

**Notebooks:**

* `check_duplicates.ipynb`: Used for data cleaning.
* `EDA&PP_Synthetic_UnityEyes.ipynb`:  Performs exploratory data analysis (EDA) and pre-processing of synthetic and Unity Eyes data.
* `*_real_eyes.ipynb`, `*_synthetic_eyes.ipynb`, `*_finetuned_real_eyes.ipynb`: Jupyter notebooks containing the training code for each model variation.

**Important Scripts:**

* `model_evaluation.py`: Contains functions to evaluate model performance on test data.
* `plot_images.py`:  Includes utilities for visualizing model outputs, including false positive predictions.
* `utils.py`:  Houses general-purpose utility functions for data manipulation and model training.

## How to Use

1. **Setup:**
   - Create an environment using `requirements.txt`: `-m venv .venv`
   - Activate the environment: `.venv/bin/activate>`
   - Install dependencies: `pip install -r requirements.txt`

2. **Data Preparation:**

3. **Model Training:**
   - Open and run the relevant Jupyter notebooks in the `models` directory to train different models (CNN, MLP, SVM).

4. **Model Evaluation:**
   - Use the `model_evaluation.py` script to evaluate the performance of trained models on your test dataset.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.