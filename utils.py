import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


def plot_confusion_matrix(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

def load_train_test_data():
    X_train = pd.read_csv('../dataset/splits/X_train.csv')
    y_train = pd.read_csv('../dataset/splits/y_train.csv').values.ravel()
    X_test = pd.read_csv('../dataset/splits/X_test.csv')
    y_test = pd.read_csv('../dataset/splits/y_test.csv').values.ravel()
    return X_train, y_train, X_test, y_test