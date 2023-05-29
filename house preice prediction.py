import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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
    
    svr = SVR(kernel='rbf', C=100)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1) 


    svr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)

    
    y_pred_svr = svr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_gbm = gbm.predict(X_test)

"""    Evaluate the models using mean squared error (MSE),
    root mean squared error (RMSE), and mean absolute percentage error (MAPE)"""
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_gbm = mean_squared_error(y_test, y_pred_gbm)

    rmse_svr = np.sqrt(mse_svr)
    rmse_rf = np.sqrt(mse_rf)
    rmse_gbm = np.sqrt(mse_gbm)

    mape_svr = np.mean(np.abs((y_test - y_pred_svr) / y_test)) * 100
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
    mape_gbm = np.mean(np.abs((y_test - y_pred_gbm) / y_test)) * 100
    

    print("Support Vector Machine:")
    print("MSE: {:.2f}".format(mse_svr))
    print("RMSE: {:.2f}".format(rmse_svr))
    print("MAPE: {:.2f}%".format(mape_svr))
    print()

    print("Random Forest:")
    print("MSE: {:.2f}".format(mse_rf))
    print("RMSE: {:.2f}".format(rmse_rf))
    print("MAPE: {:.2f}%".format(mape_rf))
    print()

    print("Gradient Boosting Machine:")
    print("MSE: {:.2f}".format(mse_gbm))
    print("RMSE: {:.2f}".format(rmse_gbm))
    print("MAPE: {:.2f}%".format(mape_gbm))
    return y_pred_svr, y_pred_rf, y_pred_gbm


def plot_results(y_test, y_pred_svr, y_pred_rf, y_pred_gbm):
    
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.scatter(y_test, y_pred_svr)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Support Vector Machine')

    plt.subplot(1,3,2)
    plt.scatter(y_test, y_pred_rf)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Random Forest')

    plt.subplot(1,3,3)
    plt.scatter(y_test, y_pred_gbm)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Gradient Boosting Machine')

    plt.show()


X,y = load_and_explore_data()
X_train,X_test,y_train,y_test = split_data(X,y)


y_pred_svr, y_pred_rf, y_pred_gbm = train_and_evaluate_models(X_train,X_test,y_train,y_test)
plot_results(y_test,y_pred_svr,y_pred_rf,y_pred_gbm)
