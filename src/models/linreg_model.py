# src/models/linreg_model.py
from sklearn.linear_model import LinearRegression
def train_linreg(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr
