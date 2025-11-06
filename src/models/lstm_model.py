# src/models/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.logger import get_logger

logger = get_logger("lstm_model")

def build_lstm_univariate(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=batch_size, callbacks=[es], verbose=0)
    return model, history
