import numpy as np
import matplotlib.pyplot as plt

# the following functions are borrowed from https://github.com/TomasBeuzen/machine-learning-tutorials/blob/master/ml-timeseries/notebooks/supervised_time_series_intro.ipynb
# Custom functions
def lag_df(df, lag, cols):
    return df.assign(
        **{f"{col}-{n}": df[col].shift(n) for n in range(1, lag + 1) for col in cols}
    )


def ts_predict(input_data, model, n=20, responses=1, lag=None):

    predictions = []
    n_features = input_data.size
    for _ in range(n):
        # make prediction
        if lag is None:
            predictions = np.append(
                predictions, model.predict(input_data.reshape(1, -1))
            )
        else:
            predictions = np.append(
                predictions, model.predict(input_data.reshape(1, 1, lag))
            )
        # new input data
        input_data = np.append(
            predictions[-responses:], input_data[: n_features - responses]
        )
    return predictions.reshape((-1, responses))


def plot_ts(ax, df_train, df_test, predictions, xlim, response_cols):
    col_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, col in enumerate(response_cols):
        ax.plot(df_train[col], "-", c=col_cycle[i], label=f"Train {col}")
        ax.plot(df_test[col], "--", c=col_cycle[i], label=f"Validation {col}")
        ax.plot(
            np.arange(
                df_train.index[-1] + 1, df_train.index[-1] + 1 + len(predictions)
            ),
            predictions[:, i],
            c=col_cycle[-i - 2],
            label=f"Prediction {col}",
        )
    ax.set_xlim(0, xlim + 1)
    ax.set_title(
        f"Train Shape = {len(df_train)}, Validation Shape = {len(df_test)}", fontsize=16
    )
    ax.set_ylabel(df_train.columns[0])


def create_rolling_features(df, columns, windows=[6, 12]):
    for window in windows:
        df["rolling_mean_" + str(window)] = df[columns].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df[columns].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df[columns].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df[columns].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df[columns].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = (
            df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        )
        df["rolling_min_max_diff_" + str(window)] = (
            df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    return df



def hms_string(sec_elapsed):
    """
    Returns the formatted time 

    Parameters:
    -----------
    sec_elapsed: int
        second elapsed

    Return:
    --------
    str
        the formatted time
    """

    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"