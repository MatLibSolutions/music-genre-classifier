import numpy as np
from models.svm_classifier import SVMModel
from models.rf_classifier import RandomForestModel
from models.knn_classifier import KNNClassifier

class EnsembleModel:
    def __init__(self):
        self.svm = SVMModel()
        self.random_forest = RandomForestModel()
        self.knn = KNNClassifier()
    
    def fit(self, X, y):
        self.svm.fit(X, y)
        self.random_forest.fit(X, y)
        self.knn.fit(X, y)
    
    def predict(self, X):
        # Get predictions from all models
        svm_pred = self.svm.predict(X)
        rf_pred = self.random_forest.predict(X)
        knn_pred = self.knn.predict(X)
        
        # Perform majority voting
        predictions = np.array([svm_pred, rf_pred, knn_pred])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
