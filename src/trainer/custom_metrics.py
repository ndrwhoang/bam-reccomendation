import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}
