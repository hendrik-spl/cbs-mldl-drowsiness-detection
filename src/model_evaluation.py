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