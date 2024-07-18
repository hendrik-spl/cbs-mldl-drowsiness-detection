# Driver Drowsiness Detection

## Overview

This repository contains code and models for a machine learning and deep learning project focused on developing a drowsiness detection system. The system utilizes various image classification models to predict whether a person's eyes are open or closed, indicating alertness or drowsiness. 
The repository is meant to be understood in the context of the accompanying report.

## Key Features

- **Multiple Model Architectures:** Explores the use of Convolutional Neural Networks (CNNs), Multi-Layer Perceptrons (MLPs), and Support Vector Machines (SVMs) for drowsiness detection.
- **Transfer Learning:** Leverages pre-trained CNN models (MobileNetV2) and fine-tuning for improved performance.
- **Synthetic Data:** Incorporates synthetic eye images to augment the training process and enhance model generalization.

## Repository Structure
    .
    ├── data
    │   ├── CEW_Data_Test_Train   (Original, unprocessed eye image datasets)
    │   └── Unity_Data_Test_Train (Cleaned, preprocessed images ready for training)
    ├── models
    │   ├── cnn (Convolutional Neural Networks - MobileNetV2)
    │   │   ├── 0_real_eyes (Models trained on real eye data)
    │   │   │   ├── last_layers (Fine-tuning only the last few layers)
    │   │   │   ├── last_layers (Fine-tuning only the last few layers)
    │   │   │   │   ├── ckpt (Checkpoints generated during model training through keras callbacks.)
    │   │   │   │   ├── history (.csv files which contain the model training history)
    │   │   │   │   ├── weights (Final Model files)
    │   │   │   │   └── *.ipynb (The respective Jupyter notebook)
    │   │   │   └── full_network (Fine-tuning the entire network)
    │   │   │   │   ├── ... (Structure similar to folder above)
    │   │   ├── 1_synthetic_eyes (Models trained on synthetic eye data)
    │   │   │   ├── last_layers
    │   │   │   └── full_network
    │   │   └── 2_fine_tuned_real_eyes (Models fine-tuned from synthetic to real)
    │   │       ├── last_layers
    │   │       └── full_network
    │   ├── mlp (Multi-Layer Perceptron models)
    │   │   ├── ... (Structure similar to CNN models)
    │   └── svm (Support Vector Machine models)
    │   │       ├── 0_real_eyes
    │   │       └── 1_data_split
    ├── src
    │   ├── utils.py (Data loading, preprocessing, etc.)
    │   ├── plot_images.py (Several image plots)
    │   └── model_evaluation.py (Evaluation scripts)
    ├── .gitignore
    ├── README.md  (This file)
    └── requirements.txt

**Notebooks:**

* `EDA&PP_*.ipynb`:  Performs exploratory data analysis (EDA) and pre-processing of synthetic and Unity Eyes data.
* `*_real_eyes.ipynb`, `*_synthetic_eyes.ipynb`, `*_finetuned_real_eyes.ipynb`: Jupyter notebooks containing the training code for each model variation.

## How to Use

1. **Setup:**
   - Create an environment using `requirements.txt`: `-m venv .venv`
   - Activate the environment: `.venv/bin/activate>`
   - Install dependencies: `pip install -r requirements.txt`

2. **Data Preparation:**
   - If the respective datasets `CEW_Data_Test_Train` and `Unity_Data_Test_Train` are available, no further action is needed. 
   - Alternatively, please contact the authors for data access.

3. **Model Training:**
   - Open the relevant Jupyter notebooks in the `models` directory and adjust the `repo_path` variable.
   - Run the notebooks to train the different models (CNN, MLP, SVM).
   - Alternatively, use `model = tf.keras.models.load_model(model_file_path, compile=True)` to load any saved model from the repository in the `weights` folders.

4. **Model Evaluation:**
   - Use the `model_evaluation.py` script to evaluate the performance of trained models on your test dataset.