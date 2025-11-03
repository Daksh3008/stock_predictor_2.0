# src/models/rf_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf
