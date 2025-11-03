# src/models/lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_univariate(input_shape, units=64, dropout=0.2):
    m = Sequential()
    m.add(Input(shape=input_shape))
    m.add(LSTM(units, activation="tanh"))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m

def train_lstm(model, Xtr, ytr, Xval, yval, epochs=30, batch_size=16):
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return model
