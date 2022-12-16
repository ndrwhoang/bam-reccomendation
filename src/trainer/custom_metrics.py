import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(p):
    (preds, _), labels = p
    # print(preds.shape)
    # print('=================')
    # print(len(labels))
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)

    return {"accuracy": accuracy}
