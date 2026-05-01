# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import backend as K

def series_to_supervised(data, train_size=0.8, n_in=7, n_out=7,
                         target_column='Global_active_power',
                         dropnan=True, scale_X=False):
    """
    Convert multivariate dataframe to supervised learning format.
    Returns: X_train, y_train, X_test, y_test, n_features
    X shape as DataFrame: (samples, n_in * n_features)
    y shape as DataFrame: (samples, n_out)
    """
    df = data.copy()
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataframe")

    features = df.copy()
    n_features = features.shape[1]

    # Build lagged X
    X_frames = []
    X_col_names = []
    for i in range(n_in, 0, -1):
        shifted = features.shift(i)
        X_frames.append(shifted)
        X_col_names += [f"{col}(t-{i})" for col in features.columns]

    # Build horizon y (only target)
    y_frames = []
    y_col_names = []
    for i in range(0, n_out):
        shifted = df[[target_column]].shift(-i)
        y_frames.append(shifted)
        y_col_names.append(f"{target_column}(t+{i})")

    X_all = pd.concat(X_frames, axis=1)
    X_all.columns = X_col_names
    y_all = pd.concat(y_frames, axis=1)
    y_all.columns = y_col_names

    supervised = pd.concat([X_all, y_all], axis=1)

    if dropnan:
        supervised.dropna(inplace=True)

    # Separate
    X = supervised.iloc[:, : n_in * n_features]
    y = supervised.iloc[:, n_in * n_features : n_in * n_features + n_out]

    # Train-test split by rows
    split_index = int(len(X) * train_size)
    X_train = X.iloc[:split_index].copy()
    y_train = y.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, y_train, X_test, y_test, n_features

# -------------------- Metrics (Keras-compatible) --------------------
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def smape(y_true, y_pred):
    # Symmetric MAPE implemented in TF
    num = K.abs(y_pred - y_true)
    den = (K.abs(y_true) + K.abs(y_pred)) + K.epsilon()
    return 100.0 * K.mean(num / den)
