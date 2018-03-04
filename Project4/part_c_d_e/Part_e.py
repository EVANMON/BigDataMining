# Project 4 -- Regression Analysis
# Part e - KNN Regression
# Author: Zhonglin Zhang

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
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
y = y.reshape(n_samples)
print(X.shape, y.shape)

def cv_rmse(X, y, kf, k, algo, w, dist):
    knn = KNeighborsRegressor(n_neighbors=k, algorithm=algo, weights=w, p=dist)
    train_rmse = []
    test_rmse = []
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        knn.fit(X_train, y_train)
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        train_rmse.append(np.mean((y_train_pred - y_train)**2))
        test_rmse.append(np.mean((y_test_pred - y_test)**2))
    return sqrt(np.mean(train_rmse)), sqrt(np.mean(test_rmse))

def select_best_par(k_list, trn_RMSE, tst_RMSE, algo, w, d):
    trn_idx = np.argmin(trn_RMSE)
    tst_idx = np.argmin(tst_RMSE)
    dist = 'Euclidean Dist' if d == 2 else 'Manhattan Dist'
    print('--------------------------------------------')
    pars = 'Algorithm: ' + algo + ' | Weights: ' + w + ' | ' + dist
    print('Regression ' + pars)
    print('For Training, the best k value is %i' % k_list[trn_idx])
    print('The best RMSE is ' + str(trn_RMSE[trn_idx]))
    print('\nFor Testing, the best k value is %i' % k_list[tst_idx])
    print('The best RMSE is ' + str(tst_RMSE[tst_idx]))
    plt.plot(k_list, trn_RMSE, label='Training')
    plt.plot(k_list, tst_RMSE, label='Testing')
    plt.title('RMSE vs k Plot (' + pars + ')')
    plt.xlabel('k value')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.savefig('pics/Algo_' + algo + '_W_' + w + '_dist_' + str(d) + '.png')
    plt.close()


kf = KFold(n_splits=10, shuffle=True, random_state=42)
k_list = range(1, 51)
algorithms = ['ball_tree', 'kd_tree', 'brute', 'auto']
weights = ['uniform', 'distance']
distances = [1, 2]

for algo in algorithms:
    for w in weights:
        for d in distances:
            print('*** For Algorithm: '+algo+', Weights: '+w+', l-p Distance: '+str(d)+ ' ***')
            train_RMSE = []
            test_RMSE = []
            for k in k_list:
                #print('For k: ', k)
                trn_err, tst_err = cv_rmse(X, y, kf, k, algo, w, d)
                train_RMSE.append(trn_err)
                test_RMSE.append(tst_err)
            select_best_par(k_list, train_RMSE, test_RMSE, algo, w, d)

