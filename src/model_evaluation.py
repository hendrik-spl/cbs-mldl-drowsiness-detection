import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

def evaluate_model(model, test_data):
    """
    Evaluate the model on the test data.
    
    Args:
    model: A compiled model.
    test_data: A tuple of test data (X_test, y_test).
    
    Returns:
    None
    """
    loss, accuracy = model.evaluate(test_data)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")