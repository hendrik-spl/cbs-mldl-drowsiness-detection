from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import time

def evaluate_model(model, test_data, labels=['Closed', 'Open'], show_cm=True, show_roc=True):
    """
    Evaluate a model on a test dataset, print metrics and plot a confusion matrix.
    
    Args:
    model: A compiled model.
    test_data: A test dataset.
    
    Returns:
    None
    """
    start_time = time.time()

    # Initialize lists to store predictions and true labels
    y_pred = []
    y_true = []
    y_pred_prob = []

    # Iterate over the test data and make predictions
    for images, labels in test_data:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=-1))
        y_true.extend(labels.numpy())
        y_pred_prob.extend(predictions[:, 1])

    duration = time.time() - start_time

    # convert predictions and true labels to numpy arrays
    y_pred = np.array(y_pred)  
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # print metrics
    print(f"Test accuracy: {round(accuracy, 3)}")
    print(f"Test precision: {round(precision, 3)}")
    print(f"Test recall: {round(recall, 3)}")
    print(f"Test F1 score: {round(f1_score, 3)}")
    print(f"Prediction time: {round(duration, 2)} seconds")

    # plot confusion matrix
    if show_cm:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Closed', 'Open'], yticklabels=['Closed', 'Open'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
    # plot ROC curve
    if show_roc:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()