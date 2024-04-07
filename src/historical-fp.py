import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neural_network

def init_data(file, features):
    fights = pd.read_csv(file)
    fights = fights.drop_duplicates()

    fights = fights.dropna(subset=features)
    data = fights[features].copy()
    data = data[data.Winner != 2]
    y = data["Winner"].to_numpy()

    data = data.drop(columns = ["Winner"])

    data = ((((data - data.min()) / (data.max() - data.min())) * 9) + 1)

    x = data.to_numpy()
    return data, x, y

def run_MLP(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    clf = neural_network.MLPClassifier(max_iter=5000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    return accuracy, confusion_matrix

def main():
    file = "fights.csv"
    features = ["Winner", "B_current_win_streak", "B_current_lose_streak", "B_longest_win_streak", "B_wins", "B_losses", "R_current_win_streak", "R_current_lose_streak", "R_longest_win_streak", "R_wins", "R_losses"]
    
    data, x, y = init_data(file, features)
    accuracy, confusion_matrix = run_MLP(x, y)

    print("Accuracy: ", accuracy)
    print("Confusion Matrix:\n", confusion_matrix)

if __name__ == "__main__":
    main()
