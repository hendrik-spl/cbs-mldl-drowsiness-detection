import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def plot_images(img_orig, img_augm, num_images=5):
    """
    Plot original and augmented images.
    
    Parameters:
        img_orig (tf.data.Dataset): Original images.
        img_augm (tf.data.Dataset): Augmented images.
        num_images (int): Number of images to plot. Default is 5.

    Returns:
        None
    """
    plt.figure(figsize=(15, 6))
    images_displayed = 0

    # Take one batch from each dataset
    for (images1, _), (images2, _) in zip(img_orig.take(1), img_augm.take(1)):
        for i in range(num_images):
            if images_displayed >= num_images:
                break

            # Plot original image
            plt.subplot(2, num_images, images_displayed + 1)
            plt.imshow(images1[i].numpy().astype("uint8"))
            plt.title(f"Original")
            plt.axis("off")

            # Plot augmented image
            plt.subplot(2, num_images, images_displayed + num_images + 1)
            plt.imshow(images2[i].numpy().astype("uint8"))
            plt.title(f"Augmented")
            plt.axis("off")

            images_displayed += 1

def plot_predictions(model, dataset, dataset_visual, class_names = ["Closed", "Open"], num_images=5):
    """
    Plot predictions.
    
    Parameters:
        model (tf.keras.Model): Model to use for predictions.
        dataset (tf.data.Dataset): Dataset with images.
        dataset_visual (tf.data.Dataset): Dataset with visual images.
        class_names (list): Class names. Default is ["Closed", "Open"].
        num_images (int): Number of images to plot. Default is 5.
    """
    plt.figure(figsize=(15, 6))
    images_displayed = 0

    for (images, labels), (images_visual, _) in zip(dataset.take(1), dataset_visual.take(1)):
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1)

        for i in range(len(images)):
            if images_displayed >= num_images:
                break

            plt.subplot(1, num_images, images_displayed + 1)
            plt.imshow(images_visual[i].numpy().astype("uint8"))
            plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[predicted_labels[i]]}")
            plt.axis("off")
            images_displayed += 1