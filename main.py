import pandas as pd
from preprocess import load_data, preprocess_data
from models.ensemble import EnsembleModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_data('data/features_30_sec.csv')

# Preprocess the data and get the label encoder
X, y, label_encoder = preprocess_data(data, fit_encoders=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the ensemble model
ensemble_model = EnsembleModel()
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
