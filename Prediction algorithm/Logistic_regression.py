from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_confusion_matrix,load_train_test_data

#Logistic regression model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

def loss_function(y, h):
    return np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

def logistic_regression(X, y, lr=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])
    losses=[]
    for epoch in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        weights -= lr * gradient
        loss = -loss_function(y, h)
        losses.append(loss)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.show()
    return weights

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_train_test_data()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    weights = logistic_regression(X_train, y_train, lr=0.01, epochs=1000)

    # Make predictions
    y_pred = predict(X_test, weights).astype(int)

    # Evaluate model
    accuracy = (y_pred == y_test).mean()
    print("Test accuracy:", accuracy)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()