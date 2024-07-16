import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def plot_false_positives(model, test_data, n=10, labels=['Closed', 'Open']):
    """
    Display n false positives from the test data set in a single plot.
    """
    plt.figure(figsize=(20, 2))
    count = 0

    for images, true_labels in test_data:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Find false positives: predicted as Open (1), but true label is Closed (0)
        false_positives_mask = (predicted_labels == 1) & (true_labels.numpy() == 0)
        
        # Iterate over potential false positives
        for i, is_false_positive in enumerate(false_positives_mask):
            if not is_false_positive:
                continue

            if count >= n:
                break

            img = images[i].numpy().squeeze()  
            img = np.clip(img, 0, 1)
            plt.subplot(1, n, count + 1)
            plt.imshow(img)
            plt.axis('off')
            count += 1
    plt.suptitle('False Positives: Predicted as Open, but True Label is Closed')
    plt.tight_layout()
    plt.show()

def plot_false_negatives(model, test_data, n=10, labels=['Closed', 'Open']):
    """
    Display n false negatives from the test data set in a single plot.
    """
    plt.figure(figsize=(20, 2))
    count = 0

    for images, true_labels in test_data:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Find false negatives: predicted as Closed (0), but true label is Open (1)
        false_negatives_mask = (predicted_labels == 0) & (true_labels.numpy() == 1)
        
        # Iterate over potential false negatives
        for i, is_false_negative in enumerate(false_negatives_mask):
            if not is_false_negative:
                continue

            if count >= n:
                break

            img = images[i].numpy().squeeze()  
            img = np.clip(img, 0, 1)
            plt.subplot(1, n, count + 1)
            plt.imshow(img)
            plt.axis('off')
            count += 1
    plt.suptitle('False Negatives: Predicted as Closed, but True Label is Open')
    plt.tight_layout()
    plt.show()