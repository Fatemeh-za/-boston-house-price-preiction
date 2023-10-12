import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_explore_data():
    boston = load_boston()
    X = boston.data 
    y = boston.target 

    df = pd.DataFrame(X, columns=boston.feature_names)
    df['MEDV'] = y 

    print(df.describe())
    df.hist(bins=10, figsize=(10,10))
    plt.show()

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR())
    ])

    param_grid = [
        {'model': [SVR()], 'model__kernel': ['rbf'], 'model__C': [0.1, 1, 10]},
        {'model': [RandomForestRegressor()], 'model__n_estimators': [10, 50, 100], 'model__max_depth': [5, 10]},
        {'model': [GradientBoostingRegressor()], 'model__n_estimators': [10, 50, 100], 'model__learning_rate': [0.01, 0.1]}
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best parameters: ", grid.best_params_)

    y_pred = grid.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("MSE: {:.2f}".format(mse))
    print("RMSE: {:.2f}".format(rmse))
    print("MAPE: {:.2f}%".format(mape))

X,y = load_and_explore_data()
X_train,X_test,y_train,y_test = split_data(X,y)
train_and_evaluate_models(X_train,X_test,y_train,y_test)
