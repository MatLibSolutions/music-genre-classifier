import os
import librosa
import numpy as np
import pandas as pd

# Function to extract MFCC features for a 30-second clip
def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# Function to extract features from 30-second or 3-second audio clips
def extract_features_from_directory(directory_path, duration=30):
    features = []
    labels = []
    genres = os.listdir(directory_path)
    
    for genre in genres:
        genre_path = os.path.join(directory_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                if file_path.endswith(".wav"):
                    mfcc_features = extract_mfcc(file_path)
                    features.append(mfcc_features)
                    labels.append(genre)
    
    return np.array(features), np.array(labels)

# Save features into CSV
def save_features_to_csv(directory_path, output_csv, duration=30):
    features, labels = extract_features_from_directory(directory_path, duration)
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)

# Example usage for 30-second clips
save_features_to_csv("data/genres_original", "data/features_30_seconds.csv", duration=30)
# Example usage for 3-second clips (split audio into 3-second chunks)
save_features_to_csv("data/genres_original", "data/features_3_seconds.csv", duration=3)
