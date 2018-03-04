# Project 4 -- Regression Analysis
# Part a, c - Neural Network
# Author: Zhonglin Zhang

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
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
X = np.array(X_df.values)
y = np.array(y_df.values)
n_samples, feature_dim = X.shape
enc = OneHotEncoder(sparse=False)
X_enc = enc.fit_transform(X)
y = y.reshape(n_samples)
print(X_enc.shape, y.shape)

# Part c: Neural Network Regression
def cv_rmse(X, y, kf, act_func, unit_n):
    reg = MLPRegressor(hidden_layer_sizes=unit_n, activation=act_func, random_state=42)
    train_rmse = []
    test_rmse = []
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        reg.fit(X_train, y_train)
        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)
        train_rmse.append(np.mean((y_train_pred - y_train)**2))
        test_rmse.append(np.mean((y_test_pred - y_test)**2))
    return sqrt(np.mean(train_rmse)), sqrt(np.mean(test_rmse))

def select_best_unit(unit_nums, RMSE, state='Testing', act='relu'):
    min_idx = np.argmin(RMSE)
    print('--------------------------------------------')
    print('For ' + state + ', the best unit number is %i' % unit_nums[min_idx])
    print('The best RMSE is ' + str(RMSE[min_idx]))
    plt.plot(unit_nums, RMSE, label=state)
    plt.title(state + ' RMSE (' + act + ')')
    plt.xlabel('Unit Number')
    plt.ylabel('RMSE')
    plt.legend()
    #plt.show()
    plt.savefig('pics/' + state + '_'+act+'.png')
    plt.close()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
unit_nums = range(1, 401, 5)
act_func = ['relu', 'logistic', 'tanh']
for act in act_func:
    train_RMSE = []
    test_RMSE = []
    print('*** For '+act+' ***')
    for n in unit_nums:
        un = tuple([n])
        trn_err, tst_err = cv_rmse(X_enc, y, kf, act, un)
        train_RMSE.append(trn_err)
        test_RMSE.append(tst_err)
    select_best_unit(unit_nums, train_RMSE, 'Training', act)
    select_best_unit(unit_nums, test_RMSE, 'Testing', act)
