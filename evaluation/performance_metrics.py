from sklearn.metrics import precision_score, recall_score, f1_score

def compute_performance_metrics(y_true, y_pred):

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1
