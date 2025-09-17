import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_confusion_matrix,load_train_test_data



def main():
    # Load data
    X_train, y_train, X_test, y_test = load_train_test_data()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    #train the model
    rf_model.fit(X_train, y_train)  # Fit the model to the training data
    y_pred = rf_model.predict(X_test)  # Predict on test data



    cm = plot_confusion_matrix(y_test, y_pred)
    # Detailed performance report
    performances=classification_report(y_test, y_pred)
    with open("../results/random_forest_results.txt", "w") as f:
        f.write(f"Confusion matrix :\n{np.array2string(cm)}\n\n")
        f.write(f"Model performances: \n {performances}\n")

if __name__ == "__main__":
    main()
