import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomZoom((-0.1, 0.1))
])

def load_and_preprocess_images(path, batch_size, image_size, seed, shuffle=False, subset=None, validation_split=None):
    """
    Load and preprocess images from a directory.

    Parameters:
        path (str): Path to the directory.
        batch_size (int): Batch size.
        image_size (tuple): Image size.
        seed (int): Random seed.
        shuffle (bool): Whether to shuffle the data.
        subset (str): Subset of the data to load.
        validation_split (float): Fraction of the data to use as validation.

    Returns:
        tf.data.Dataset: Preprocessed images.
    """
    raw_data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=shuffle,
        subset=subset,
        validation_split=validation_split,
    )

    # # Apply data augmentation only if subset is 'training'
    # if subset == 'training':
    #     raw_data = raw_data.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Apply MobileNet preprocessing
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    preprocessed_data = raw_data.map(lambda x, y: (preprocess_input(x), y))

    return preprocessed_data.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)