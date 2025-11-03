import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def make_lstm_sequences(data, lookback):
    """
    Convert a 1D array into supervised LSTM format (X, y).
    Example: For lookback=10, each X is previous 10 timesteps and y is the next value.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y

def build_lstm_univariate(input_shape):
    """
    Build a simple univariate LSTM network.
    input_shape: (lookback, 1)
    """
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
    """
    Train the given LSTM model with early stopping.
    """
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es]
    )
    return model, history
