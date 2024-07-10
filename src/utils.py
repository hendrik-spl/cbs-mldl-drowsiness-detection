import tensorflow as tf

def load_and_preprocess_images(path, batch_size, image_size, seed, data_aug_rate=0, shuffle=True, subset=None, validation_split=None, skip_preprocessing=False):
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

    tf.keras.utils.set_random_seed(seed)

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