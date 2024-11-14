import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """
    Load the CSV dataset and separate features from labels.
    """
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns like 'filename'
    df = df.drop(columns=['filename'])
    
    return df

def preprocess_data(df, fit_encoders=True):
    """
    Preprocess the dataset:
    - Handle missing values (if any)
    - Encode categorical labels
    - Normalize features
    """
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Convert target labels to numerical values
    if fit_encoders:
        y = label_encoder.fit_transform(y)
    else:
        y = label_encoder.transform(y)

    # Normalize the features
    if fit_encoders:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, label_encoder
