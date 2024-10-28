import time
import warnings

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

start = time.time()

import numpy as np

np.random.seed(42)


def scoring(model, x_test, y_test) -> tuple:
    ps = precision_score(y_test, model.predict(x_test))
    rs = recall_score(y_test, model.predict(x_test))
    fs = f1_score(y_test, model.predict(x_test))
    mcc = matthews_corrcoef(y_test, model.predict(x_test))
    return ps, rs, fs, mcc


def make_table():
    return pd.DataFrame(
        columns=[
            "Score",
            "Precision _score",
            "Recall_score",
            "F1_score",
            "Matthews_Corr_Coef",
        ]
    )


df = pd.read_csv(
    "/home/david/Documents/ARU/AvalancheProject/demo/load/balanced_cleaned.csv",
    usecols=[
        "Mean Temp (°C)",
        "Total Rain (mm)",
        "Total Snow (cm)",
        "Total Precip (mm)",
        "Snow on Grnd (cm)",
        "avalanche",
    ],
)

X = df[df.columns[:-1]]
y = df[df.columns[-1]]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

estimators = [
    ("minmaxscaler", MinMaxScaler()),
    ("standardscaler", StandardScaler()),
    ("clf", RandomForestClassifier(max_depth=4, max_features=0.75, n_estimators=30)),
]

pipe = Pipeline(estimators)

pipe.fit(X_train, y_train)

score = pipe.score(X_train, y_train)
ps, rs, fs, mcc = scoring(pipe, X_train, y_train)


print("Training results")
df = make_table()
results = [score, ps, rs, fs, mcc]
df.loc[len(df)] = results
print(tabulate(df, headers="keys", tablefmt="psql"))


score = pipe.score(X_test, y_test)
ps, rs, fs, mcc = scoring(pipe, X_test, y_test)


print("Testing results")
df = make_table()
results = [score, ps, rs, fs, mcc]
df.loc[len(df)] = results
print(tabulate(df, headers="keys", tablefmt="psql"))

time_taken = time.time() - start
print("Time to taken for demo_pipe.py to run: {}s.".format(round(time_taken, 3)))
