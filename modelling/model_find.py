"""
Main script. Iterates through models found in MODELS.
The main loop considers both raw data and nomalised data.
Results are exported to csv files "simple_data.csv" and norm_data.csv" respectively.

David O'Connell
do363@student.aru.ac.uk
"""

import scoring
import data_prep

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")


start = time.time()

CSV = "../load/balanced_cleaned.csv"

MODELS = [
    NearestCentroid(),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(n_estimators=5),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=50),
    RandomForestClassifier(n_estimators=500),
    RandomForestClassifier(n_estimators=1000),
    SVC(kernel="linear", C=1.0),
    SVC(kernel="rbf", C=1.0, gamma=0.03333),
]


def run(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    model,
):
    start_time = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time
    start_time = time.time()
    score = model.score(X_test, Y_test)
    test_time = time.time() - start_time
    return train_time, test_time, score


def train(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    df: pd.DataFrame,
    models=MODELS,
) -> pd.DataFrame:
    df = df
    for model in models:
        train_time, test_time, score = run(X_train, Y_train, X_test, Y_test, model)
        ps, rs, fs, mcc = scoring.scoring(model, X_test, Y_test)

        results = [str(model), train_time, test_time, score, ps, rs, fs, mcc]
        df.loc[len(df)] = results
    return df


def main():
    df = data_prep.make_table()
    X_train, Y_train, X_test, Y_test = data_prep.split_data(data_prep.load_data(CSV))

    print("Models trained on raw data:")
    simple_data = train(X_train, Y_train, X_test, Y_test, df)
    print(tabulate(simple_data, headers="keys", tablefmt="psql"))
    simple_data.to_csv("./results.csv", index=False)

    df = data_prep.make_table()
    X_train, Y_train, X_test, Y_test = data_prep.split_data(
        data_prep.min_max(data_prep.load_data(CSV))
    )  # noqa: E501
    print("Models trained on min_max data:")
    scaled_data = train(X_train, Y_train, X_test, Y_test, df)
    print(tabulate(scaled_data, headers="keys", tablefmt="psql"))
    scaled_data.to_csv("./min_max_results.csv", index=False)

    df = data_prep.make_table()
    X_train, Y_train, X_test, Y_test = data_prep.split_data(
        data_prep.pca(data_prep.load_data(CSV))
    )  # noqa: E501
    print("Models trained on pca data:")
    scaled_data = train(X_train, Y_train, X_test, Y_test, df)
    print(tabulate(scaled_data, headers="keys", tablefmt="psql"))
    scaled_data.to_csv("./pca_results.csv", index=False)

    df = data_prep.make_table()
    X_train, Y_train, X_test, Y_test = data_prep.split_data(
        data_prep.std_scale(data_prep.load_data(CSV))
    )  # noqa: E501
    print("Models trained on scaled data:")
    scaled_data = train(X_train, Y_train, X_test, Y_test, df)
    print(tabulate(scaled_data, headers="keys", tablefmt="psql"))
    scaled_data.to_csv("./std_scale_results.csv", index=False)


main()
time_taken = time.time() - start
print("Time to taken for model_find.py to run: {}s.".format(round(time_taken, 3)))
