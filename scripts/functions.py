import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import datetime
import scipy.stats as stats
from scipy.interpolate import interp1d
from tqdm import tqdm
from time import time

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import pmdarima
from pmdarima.arima import ndiffs

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_validate,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

from borrowed_functions import lag_df, ts_predict, plot_ts, hms_string

def split_data(data, split_ratio=0.95, num=30):

    train_valid, test = (
        data.iloc[: - num],
        data.iloc[- num:],
    )
    splits = int(train_valid.shape[0] * split_ratio)
    train, valid = (
        train_valid.iloc[: - num], 
        train_valid.iloc[- num:]
    )

    return train, valid, test, train_valid

# class Splitter():
#     def __init__(self, n_splits=5, n=30):
#         self.n_splits = n_splits
#         self.n = n

#     def split(self, data, y=None, groups=None):
#         splits = range(data.shape[0])
#         splits = np.array_split(np.array(splits), self.n_splits)
#         splits = [(l[:- self.n], l[- self.n :]) for l in splits]
#         for X, y in splits:
#             yield X, y

#     def get_n_splits(self, X=None, y=None, groups=None):
#         return self.n_splits

def plot_forecast(train, obs, fc_series, title, lower_series=None, upper_series=None):
    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train.close, label="training")
    plt.plot(obs.close, label="observation")
    plt.plot(fc_series, label="prediction")
    if lower_series is not None:
        plt.fill_between(
            lower_series.index, lower_series, upper_series, color="k", alpha=0.15
        )
    plt.title(f"{title} Observation vs Prediction")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()
    plt.savefig(f'../results/{title}.png', bbox_inches='tight')


###############################################################################
####################### Functions for ARIMA models ############################
###############################################################################

def arima_predict(model_fit, num, index, auto=False):

    if not auto:
        # Forecast with 95% conf
        fc, se, conf = model_fit.forecast(num, alpha=0.05)

    else:
        model_fit.plot_diagnostics(figsize=(10, 10))
        plt.show()

        # Forecast
        try:
            fc, conf = model_fit.predict(n_periods=int(num), return_conf_int=True)
        except:
            fc = model_fit.predict(index[0], index[-1])
            
    # Make as pandas series
    fc_series = pd.Series(fc, index=index)
    try:
        lower_series = pd.Series(conf[:, 0], index=index)
        upper_series = pd.Series(conf[:, 1], index=index)
    except:
        lower_series = None
        upper_series = None
    return fc_series, lower_series, upper_series


def evaluate_model(pred, obs, index):
    scores = {}

    # Mean Absolute Percentage Error
    scores["MAPE"] = [np.mean(np.abs(pred - obs) / np.abs(obs))]

    # Root Mean Squared Error
    scores["RMSE"] = [np.mean((pred - obs) ** 2) ** 0.5]

    mins = np.amin(np.hstack([pred[:, None], obs[:, None]]), axis=1)
    maxs = np.amax(np.hstack([pred[:, None], obs[:, None]]), axis=1)
    scores["Min-Max Error"] = [1 - np.mean(mins / maxs)]

    return pd.DataFrame(scores, index=[index])


def plot_trans_train(trans_train, trans):

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(trans_train.close, label="transformed train")

    plt.title(f"Train after differencing and {trans} transformation")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()

###############################################################################
############## Functions for classic supervised learning models ###############
###############################################################################

def cross_validation(key, n_splits, response_col, trans_train, param_grid):

    regressor = {
        "RandomForestRegressor": RandomForestRegressor(random_state=SEED),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=SEED),
        "LGBMRegressor": LGBMRegressor(random_state=SEED),
    }
    model = regressor[key]
#     tscv = Splitter(n_splits=n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    l = []
    cv_mean = []
    cv_std = []

    for lag in range(1, 21):

        df_lag = lag_df(trans_train, lag, response_col).dropna()

        cv_score = cross_validate(
            model,
            df_lag.drop(columns=response_col),
            df_lag[response_col[0]],
            cv=tscv,
            scoring="neg_root_mean_squared_error",
        )

        l.append(lag)
        cv_mean.append(round(cv_score["test_score"].mean(), 3))
        cv_std.append(round(cv_score["test_score"].std(), 3))

    results = (
        pd.DataFrame({"lag": l, "cv_mean": cv_mean, "cv_std": cv_std})
        .set_index("lag")
        .sort_values(by="cv_mean", ascending=False)
        .head(5)
    )
    print(results)

    lag = results.index[0]
    df_lag = lag_df(trans_train, lag, response_col).dropna()

    model = GridSearchCV(
        model, param_grid, scoring="neg_root_mean_squared_error", cv=tscv
    )

    model.fit(df_lag.drop(columns=response_col), df_lag[response_col[0]])
    print(f"The best hyperparameters when lag = {lag}:\n{model.best_params_}")
    return model, lag


def regressor_predict(
    model, train_trans, lag, response_col, index, start, log=True, lamb=None
):

    df_lag = lag_df(train_trans, lag, response_col).dropna()

    # starting data for first prediction
    input_data = df_lag.iloc[-1, : lag].to_numpy()

    predictions = ts_predict(input_data, model, len(index))
    if log:
        predict = start * np.exp(np.cumsum(predictions))
    else:
        predict = np.exp(np.log(lamb * (start + np.cumsum(predictions)) + 1) / lamb)
    return pd.Series(predict, index=index)


def analyze_regressor(
    regressor,
    title,
    train,
    test,
    n_splits,
    response_col,
    trans_train,
    param_grid,
    index,
    start,
    log=True,
    lamb=None,
):
    print(
        f"Performing cross-validation to optimzie the lag and",
        f"hyperparameters for the {regressor} regressor ...",
    )
    model, lag = cross_validation(
        regressor, n_splits, response_col, trans_train, param_grid
    )

    predict = regressor_predict(
        model, trans_train, lag, response_col, index, start, log=log, lamb=lamb,
    )

    plot_forecast(train, test, predict, title)
    scores = evaluate_model(predict, test.squeeze(), title)
    return model, scores


###############################################################################
########################## Functions for LSTM models ##########################
###############################################################################

def reshape_data(data):

    X = np.reshape(data[:, 1:], (data.shape[0], 1, data.shape[1] - 1))
    return X, data[:, 0]


def train_lstm(train_X, train_y, lag, epochs, verbose=0):
    # create and fit the LSTM network
    model = Sequential()
#     model.add(Dense(lag + 1, activation="relu"))
    model.add(LSTM(lag + 1, input_shape=(1, lag)))
#     model.add(Dense(1, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(train_X, train_y, epochs=epochs, batch_size=1, verbose=verbose)

    return model


def lstm_predict(start, model, scaler, index, lag, n=30, responses=1):

    predict = ts_predict(start, model, n=n, responses=responses, lag=lag)
    # invert predictions
    predict = scaler.inverse_transform(predict)

    fc_series = pd.Series(predict.flatten(), index=index)
    return fc_series


def get_inserved_data(data, scaler):
    inversed = data[["close"]].copy()
    inversed["close"] = scaler.inverse_transform(data)
    return inversed


def lstm_results(train, test, lag, response_col, scaler, title):

    train_X, train_y = reshape_data(train.to_numpy())
    test_X, test_y = reshape_data(test.to_numpy())
    model = train_lstm(train_X, train_y, lag, epochs=30, verbose=2)

    inversed_test = get_inserved_data(test, scaler)
    inversed_train = get_inserved_data(train, scaler)
    predict = lstm_predict(
        train_X[-1].flatten(),
        model,
        scaler,
        test.index,
        lag,
        n=test.shape[0],
        responses=1,
    )
    plot_forecast(inversed_train, inversed_test, predict, title)
    score = evaluate_model(predict, inversed_test.squeeze(), title)
    return model, score


def lstm_cross_validation(data, response_col, scaler, n=21):
    l = []
    cv_mean = []
    cv_std = []
    for lag in tqdm(range(1, n)):
        print(f"lag = {lag}")
        df_lag = lag_df(data, lag, response_col).dropna()
        X, y = reshape_data(df_lag.to_numpy())
#         tscv = Splitter(n_splits=5)
        tscv = TimeSeriesSplit(n_splits=5)
        cv = []
        for train_index, test_index in tscv.split(train):

            train_X, train_y = X[train_index, :], y[train_index]
            valid_X, valid_y = X[test_index, :], y[test_index]
            model = train_lstm(train_X, train_y, lag, epochs=30)

            inversed_valid = scaler.inverse_transform(valid_y.reshape(-1, 1))
            predict = lstm_predict(
                train_X[-1].flatten(),
                model,
                scaler,
                None,
                lag,
                n=valid_X.shape[0],
                responses=1,
            )
            cv.append(np.sqrt(mean_squared_error(predict, inversed_valid.squeeze())))

        l.append(lag)
        cv_mean.append(round(np.mean(cv), 3))
        cv_std.append(round(np.std(cv), 3))

    return (
        pd.DataFrame({"lag": l, "cv_mean": cv_mean, "cv_std": cv_std})
        .set_index("lag")
        .sort_values(by="cv_mean", ascending=True)
        .head(10)
    )


