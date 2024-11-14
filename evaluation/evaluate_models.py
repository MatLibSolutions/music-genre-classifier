from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
   
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    return accuracy, cm
