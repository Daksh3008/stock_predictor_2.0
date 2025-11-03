
from sklearn.linear_model import LinearRegression
from src.models.trainer import save_model_artifact, make_artifact_path
from src.utils.logger import get_logger

logger = get_logger("linreg_model")

def train_linreg(X_train, y_train, ticker="UNKNOWN"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    try:
        path = make_artifact_path("linreg", ticker)
        metadata = {
            "model": "linear_regression",
            "ticker": ticker,
            "train_rows": int(X_train.shape[0])
        }
        save_model_artifact(lr, path, metadata)
    except Exception as e:
        logger.warning(f"Failed to save linreg artifact: {e}")
        path = None
    return lr, path
