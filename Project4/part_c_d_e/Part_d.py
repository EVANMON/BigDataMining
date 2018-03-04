# Project 4 -- Regression Analysis
# Part d - Polynomial Function
# Author: Zhonglin Zhang

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from math import sqrt

day2num = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
           'Thursday': 4, 'Friday': 5, 'Saturday': 6,
           'Sunday': 7}

# Part a
def read_data(filename):
    df = pd.read_csv(filename, delimiter=',')
    return df

def get_workflowID(str):
    fix_part = 'work_flow_'
    return int(str[len(fix_part):])

def get_fileNum(str):
    fix_part = 'File_'
    return int(str[len(fix_part):])

#Loading dataset (type: DataFrame)
dataset = read_data('network_backup_dataset.csv')
columns = list(dataset.columns.values)
dataset['Day of Week'] = dataset['Day of Week'].apply(lambda x: day2num[x])
dataset['Work-Flow-ID'] = dataset['Work-Flow-ID'].apply(lambda x: get_workflowID(x))
dataset['File Name'] = dataset['File Name'].apply(lambda x: get_fileNum(x))
X_df = dataset.drop(['Size of Backup (GB)', 'Backup Time (hour)'], axis=1)
y_df = dataset.loc[:, dataset.columns == 'Size of Backup (GB)']

X_dict = {}
y_dict = {}
for wf in X_df['Work-Flow-ID']:
    if wf not in X_dict.keys():
        y_dict[wf] = np.array(y_df.loc[dataset['Work-Flow-ID'] == wf].values)
        X_wf = X_df.loc[dataset['Work-Flow-ID'] == wf]
        X_wf = X_wf.drop(['Work-Flow-ID'], axis=1)
        X_dict[wf] = np.array(X_wf.values)
        n, _ = X_dict[wf].shape
        y_dict[wf] = y_dict[wf].reshape(n)

def cv_rmse(X, y, kf, deg):
    pf = PolynomialFeatures(deg)
    lr = LinearRegression(fit_intercept=True)
    model = make_pipeline(pf, lr)
    train_rmse = []
    test_rmse = []

    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_rmse.append(np.mean((y_train_pred - y_train)**2))
        test_rmse.append(np.mean((y_test_pred - y_test)**2))
    return sqrt(np.mean(train_rmse)), sqrt(np.mean(test_rmse))

def select_best_deg(degs, trn_RMSE, tst_RMSE, wf):
    trn_idx = np.argmin(trn_RMSE)
    tst_idx = np.argmin(tst_RMSE)
    print('--------------------------------------------')
    print('Work Flow ID: ' + str(wf))
    print('For Training, the best degree is %i' % degs[trn_idx])
    print('The best RMSE is ' + str(trn_RMSE[trn_idx]))
    print('\nFor Testing, the best degree is %i' % degs[tst_idx])
    print('The best RMSE is ' + str(tst_RMSE[tst_idx]))
    plt.plot(degs, trn_RMSE, label='Training')
    plt.plot(degs, tst_RMSE, label='Testing')
    plt.title('RMSE vs Degree Plot (WorkFlow ' + str(wf) + ')')
    plt.xlabel('Degree')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.savefig('pics/WorkFlow_' + str(wf) + '.png')
    plt.close()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
degrees = range(1, 21, 2)
for wf in X_dict:
    print('*** For Work Flow '+str(wf)+' ***')
    X_ = X_dict[wf]
    y_ = y_dict[wf]
    train_RMSE = []
    test_RMSE = []
    for deg in degrees:
        print('For degree: ', deg)
        trn_err, tst_err = cv_rmse(X_, y_, kf, deg)
        train_RMSE.append(trn_err)
        test_RMSE.append(tst_err)
    select_best_deg(degrees, train_RMSE, test_RMSE, wf)
