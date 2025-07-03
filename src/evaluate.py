import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from src.data_loader import load_data
from src.model import build_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def main():
    print("🔎 Running evaluation…")
    data_dir = 'data/processed'
    batch_size = 32

    val_ds = load_data(os.path.join(data_dir, 'val'),
                       batch_size=batch_size,
                       shuffle=False)

    model = build_model(num_classes=2)
    model.load_weights('models/best_weights.weights.h5')

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=['up', 'down']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.xticks([0,1], ['up','down'])
    plt.yticks([0,1], ['up','down'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
