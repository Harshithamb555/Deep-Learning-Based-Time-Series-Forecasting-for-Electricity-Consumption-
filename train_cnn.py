# train_cnn.py
import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import series_to_supervised
from get_dataset import CSV_PATH

with open("parameters.json") as f:
    params = json.load(f)
LOOK_BACK = int(params.get("look_back", 7))
HORIZON = int(params.get("horizon", 7))
TRAIN_SIZE = 0.8
TARGET = "Global_active_power"

os.makedirs("checkpoint", exist_ok=True)
checkpoint_path = os.path.join("checkpoint", "CNN_10F.h5")

df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime").sort_index()
raw_cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df = df[raw_cols]
df['day'] = df.index.day
df['weekday'] = ((df.index.dayofweek) // 5 == 1).astype(float)
df['season'] = [m % 12 // 3 + 1 for m in df.index.month]
df.dropna(inplace=True)

X_train, y_train, X_test, y_test, N_FEATURES = series_to_supervised(
    df, train_size=TRAIN_SIZE, n_in=LOOK_BACK, n_out=HORIZON, target_column=TARGET, dropnan=True
)

X_train_np = X_train.values.reshape((-1, LOOK_BACK, N_FEATURES))
X_test_np = X_test.values.reshape((-1, LOOK_BACK, N_FEATURES))
y_train_np = y_train.values
y_test_np = y_test.values

print("CNN shapes:", X_train_np.shape, y_train_np.shape, X_test_np.shape, y_test_np.shape)

model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=(LOOK_BACK, N_FEATURES)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation="relu", padding="same"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(HORIZON)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

model.fit(X_train_np, y_train_np, validation_data=(X_test_np, y_test_np),
          epochs=50, batch_size=32, callbacks=callbacks, verbose=1)

print("Saved CNN to", checkpoint_path)
