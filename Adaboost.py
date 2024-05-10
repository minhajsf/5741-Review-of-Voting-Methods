import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def get_error_rate(pred: np.ndarray, y: np.ndarray) -> float:
    return np.mean(pred != y)


def plot_error_rate(er_train: list[float], er_test: list[float], num_classifiers) -> None:
    df_error = pd.DataFrame({'Training': er_train, 'Test': er_test})
    ax = df_error.plot(linewidth=1.5, figsize=(10, 6), color=['lightblue', 'darkblue'], grid = False)
    ax.set_xlabel('# classifiers', fontsize=12)
    ax.set_ylabel('Error %', fontsize=12)
    ax.axhline(y=er_test[0], linewidth=1, color='red', linestyle='dashed')
    ax.legend(['Training Error', 'Test Error', 'Initial Test Error'])  # Add the legend
    plt.xticks(range(0, num_classifiers, 100))
    plt.savefig('plot.jpeg')
    plt.show()

def plot_nerror_rate(er_train: list[float], er_test: list[float], num_classifiers) -> None:
    df_error = pd.DataFrame({'Training': er_train, 'Test': er_test})
    ax = df_error.plot(linewidth=1.5, figsize=(10, 6), color=['lightblue', 'darkblue'], grid = False)
    ax.set_xlabel('# classifiers', fontsize=12)
    ax.set_ylabel('Error %', fontsize=12)
    ax.axhline(y=er_test[0], linewidth=1, color='red', linestyle='dashed')
    ax.legend(['Training Error', 'Test Error', 'Initial Test Error'])  # Add the legend
    plt.xticks(range(0, num_classifiers, 1_000))
    plt.savefig('plot.jpeg')
    plt.show()

if __name__ == '__main__':
    classifiers = 5_000
    sample_size = 10_000

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist['data'], mnist['target']
    y = y.astype(int)
    indices = np.random.choice(range(len(X)), size=sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    base_clf = DecisionTreeClassifier(max_depth=5, random_state=1)

    clf_ada = AdaBoostClassifier(base_estimator=base_clf, n_estimators=classifiers, random_state=42)
    clf_ada.fit(X_train, y_train)
    staged_train_errors = []
    staged_test_errors = []
    for i, (train_pred, test_pred) in enumerate(zip(clf_ada.staged_predict(X_train), clf_ada.staged_predict(X_test)), start=1):
        train_error = get_error_rate(train_pred, y_train)
        test_error = get_error_rate(test_pred, y_test)
        staged_train_errors.append(train_error)
        staged_test_errors.append(test_error)

        if i % 50 == 0 or i == 1:  
            print(f"Iteration {i:3d}: Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    final_test_pred = clf_ada.predict(X_test)
    print("\nFinal Classification Report on Test Set:\n", classification_report(y_test, final_test_pred))
    initial_train_error = 1 - accuracy_score(y_train, clf_ada.predict(X_train))
    initial_test_error = 1 - accuracy_score(y_test, final_test_pred)
    staged_train_errors.insert(0, initial_train_error)
    staged_test_errors.insert(0, initial_test_error)
    plot_error_rate(staged_train_errors, staged_test_errors, classifiers)

    plot_nerror_rate(staged_train_errors[10:], staged_test_errors[10:], classifiers[10:])