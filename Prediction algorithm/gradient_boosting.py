from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import sys
import os
from sklearn import svm
from sklearn.model_selection import GridSearchCV
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_train_test_data, plot_confusion_matrix

def main():
    X_train, y_train, X_test, y_test = load_train_test_data()

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #grid search 

    model = GradientBoostingClassifier()
    param_grid = {'n_estimators': [1, 10, 100, 1000], 'learning_rate': [0.1, 0.2, 0.3], 'max_depth': [3, 4, 5]}
    grid_search= GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    # Using the best parameters from grid search
    best_params = grid_search.best_params_

    # Create the model
    gb_model = GradientBoostingClassifier(**best_params, random_state=42)
    # Train the model
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)

    # Confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.title('Confusion Matrix - Gradient Boosting')

    performances= classification_report(y_test, y_pred)
    print(performances)
    with open("../results/gradient_boosting_results.txt", "w") as f:
        f.write(f"Confusion matrix :\n{np.array2string(cm)}\n\n")
        f.write(f"Model performances: \n {performances}\n")

if __name__ == "__main__":
    main()