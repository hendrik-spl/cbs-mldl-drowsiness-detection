import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def load_and_preprocess_images(path, batch_size, image_size, seed, data_aug_rate=0, shuffle=True, subset=None, validation_split=None, skip_preprocessing=False):
    """
    Load and preprocess images from a directory.

    Parameters:
        path (str): Path to the directory.
        batch_size (int): Batch size.
        image_size (tuple): Image size.
        seed (int): Random seed.
        data_aug_rate (float): Data augmentation rate. Default is 0.
        shuffle (bool): Whether to shuffle the data. Default is True.
        subset (str): Subset of the data to load. Default is None.
        validation_split (float): Fraction of the data to use as validation. Default is None.
        skip_preprocessing (bool): Whether to skip MobileNet preprocessing. Default is False.

    Returns:
        tf.data.Dataset: Preprocessed images.
    """

    tf.keras.utils.set_random_seed(seed)

    if subset == 'training' and data_aug_rate == 0:
        print('Warning: data_aug_rate is 0, but subset is set to training. No data augmentation will be applied.')

    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal', seed=seed),
    tf.keras.layers.RandomRotation(data_aug_rate, seed=seed),
    tf.keras.layers.RandomContrast(data_aug_rate, seed=seed),
    tf.keras.layers.RandomZoom((-data_aug_rate, data_aug_rate), seed=seed),
    tf.keras.layers.RandomBrightness(data_aug_rate, seed=seed),
    tf.keras.layers.GaussianNoise(data_aug_rate, seed=seed)
    ])

    raw_data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=shuffle,
        subset=subset,
        validation_split=validation_split,
    )

    # Apply data augmentation only if subset is 'training'
    if subset == 'training':
        preprocessed_data = raw_data.map(lambda x, y: (data_augmentation(x, training=True), y))
    else:
        preprocessed_data = raw_data

    # Apply MobileNet preprocessing
    if not skip_preprocessing:
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        preprocessed_data = preprocessed_data.map(lambda x, y: (preprocess_input(x), y))

    return preprocessed_data.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def plot_history(comment, history):
    """
    Plot training history.
    
    Parameters:
        comment (str): Comment to display in the plot.
        history (tf.keras.callbacks.History): Training history.

    Returns:
        None
    """
    plt.figure(figsize=(14, 4))
    plt.suptitle(comment)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    print(f'Best train_accuracy: {np.max(history.history["accuracy"]).round(4)}')
    print(f'Best train_loss: {np.min(history.history["loss"]).round(4)}')
    print(f'Best val_accuracy: {np.max(history.history["val_accuracy"]).round(4)}')
    print(f'Best val_loss: {np.min(history.history["val_loss"]).round(4)}')
    print(f'Last improvement at epoch: {np.argmax(history.history["val_accuracy"])+1}')

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".jpg"):
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
    return images, labels