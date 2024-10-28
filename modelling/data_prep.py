"""
Helper functions for loading preparing data for training and testing.

David O'Connell
do363@student.aru.ac.uk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data(path_to_csv) -> pd.DataFrame:
    df = pd.read_csv(
        path_to_csv,
        usecols=[
            "Mean Temp (°C)",
            "Total Rain (mm)",
            "Total Snow (cm)",
            "Total Precip (mm)",
            "Snow on Grnd (cm)",
            "avalanche",
        ],
    )
    return df


def split_data(data: pd.DataFrame, ratio: float = 0.3) -> tuple:
    train, test = train_test_split(data, test_size=ratio, random_state=42)
    X_train = train[train.columns[:-1]]
    Y_train = train[train.columns[-1]]
    X_test = test[train.columns[:-1]]
    Y_test = test[train.columns[-1]]
    return X_train, Y_train, X_test, Y_test


def min_max(data: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    labels = data[data.columns[-1]]
    df = data[data.columns[:-1]]
    scaled = scaler.fit_transform(df)
    return pd.concat([pd.DataFrame(scaled, columns=df.columns), labels], axis=1)


def pca(data: pd.DataFrame) -> pd.DataFrame:
    pca = PCA()
    labels = data[data.columns[-1]]
    df = data[data.columns[:-1]]
    scaled = pca.fit_transform(df)
    return pd.concat([pd.DataFrame(scaled, columns=df.columns), labels], axis=1)


def std_scale(data: pd.DataFrame) -> pd.DataFrame:
    std = StandardScaler()
    labels = data[data.columns[-1]]
    df = data[data.columns[:-1]]
    scaled = std.fit_transform(df)
    return pd.concat([pd.DataFrame(scaled, columns=df.columns), labels], axis=1)


def make_table():
    return pd.DataFrame(
        columns=[
            "Model",
            "Train_time",
            "Test_time",
            "Score",
            "Precision _score",
            "Recall_score",
            "F1_score",
            "Matthews_Corr_Coef",
        ]
    )
