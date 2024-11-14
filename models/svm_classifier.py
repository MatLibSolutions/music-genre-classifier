from sklearn.svm import SVC

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel='linear', random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
