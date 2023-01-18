import numpy as np
import datetime
import yfinance as yf
import pandas as pd


def prepare_data(X, Y):
    X = pd.DataFrame(X)
    X.columns = ["day_" + str(i) for i in range(14)]
    Y = pd.DataFrame(Y)
    Y.columns = ["to_predict"]
    df = pd.concat([X, Y], axis=1)

    train_data, test_data = train_test_split(df, test_size=0.25, random_state=4284)
    return train_data, test_data


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def digitize(n):
    if n > 0:
        return 1
    return 0


def prepare_training_data(data):
    """
    Return a prepared numpy dataframe
    input : Dataframe with expected schema

    """
    data["Delta"] = data["Close"] - data["Open"]
    data["to_predict"] = data["Delta"].apply(lambda d: digitize(d))
    return data


def acquire_training_data():
    yf.pdr_override()
    y_symbols = ["BTC-USD"]

    startdate = datetime.datetime(2022, 1, 1)
    enddate = datetime.datetime(2022, 12, 31)
    df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)
    return df


if __name__ == "__main__":
    training_data = acquire_training_data()
    prepared_training_data_df = prepare_training_data(training_data)
    btc_mat = prepared_training_data_df.to_numpy()
    WINDOW_SIZE = 14
    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = prepared_training_data_df["to_predict"].to_numpy()[WINDOW_SIZE:]
    train_data, test_data = prepare_data(X, Y)
