import pandas as pd

def load_data_from_csv(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    

    df['genre'] = df['filename'].apply(lambda x: x.split('.')[0]) 
    
    # Now extract features and labels
    X = df.drop(['genre', 'filename'], axis=1)  # Remove non-feature columns
    y = df['genre']  # This is the genre column now
    
    return X, y
