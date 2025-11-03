#after training, save the keras model and metadata

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from src.models.trainer import save_model_artifact, make_artifact_path
from src.utils.logger import get_logger
import os

logger = get_logger("lstm_model")

def build_lstm_univariate(input_shape, units=64, dropout=0.2):
    m = Sequential()
    m.add(Input(shape=input_shape))
    m.add(LSTM(units, activation="tanh"))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m

def train_lstm(model, Xtr, ytr, Xval, yval, epochs=30, batch_size=16, ticker="UNKNOWN"):
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    # Save keras model (h5)
    try:
        path = make_artifact_path("lstm", ticker)
        # ensure keras saves to .h5 file
        h5path = path.replace(".pkl", ".h5")
        model.save(h5path)
        metadata = {
            "model": "lstm",
            "ticker": ticker,
            "train_rows": int(Xtr.shape[0]),
            "epochs": int(epochs)
        }
        # write metadata json
        base = h5path.replace(".h5", "")
        meta_path = base + ".meta.json"
        import json
        with open(meta_path, "w") as f:
            json.dump(metadata, f, default=str, indent=2)
        logger.info(f"Saved LSTM model to {h5path} and metadata to {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to save LSTM artifact: {e}")
        h5path = None
    return model, h5path
