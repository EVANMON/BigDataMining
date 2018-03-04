# Project 4 -- Regression Analysis
# Part a - Dataset Loading
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
#print(dataset.shape)
wf_bs = {}
##Dictionary: Key: WorkFlow ID (type: int)
##            Value: Backup Sizes (type: list of int)
for _, row in dataset.iterrows():
    week = row['Week #']
    day = row['Day of Week']
    date = int(7 * (week-1) + (day-1))
    wf_id = int(row['Work-Flow-ID'])
    size = row['Size of Backup (GB)']

    if wf_id not in wf_bs.keys():
        wf_bs[wf_id] = np.zeros(date+1)
        wf_bs[wf_id][date] = size
    else:
        #print(date, len(wf_bs[wf_id]), wf_id)
        if date == len(wf_bs[wf_id]):
            wf_bs[wf_id] = np.append(wf_bs[wf_id], size)
        elif date < len(wf_bs[wf_id]):
            wf_bs[wf_id][date] += size
        else:
            for i in range(len(wf_bs[wf_id]), date):
                wf_bs[wf_id] = np.append(wf_bs[wf_id], 0)
            wf_bs[wf_id] = np.append(wf_bs[wf_id],size)

def plot_backup(dict, start, end, title):
    x = range(start, end+1)
    for wf in range(0, 5):
        y = dict[wf][start:end+1]
        #print(wf, y)
        plt.plot(x, y, label='WorkFlow '+str(wf))
    plt.title(title)
    plt.xlabel('Day Number')
    plt.ylabel('Backup Size (GB)')
    plt.legend(loc='best')
    plt.savefig('pics/'+ title + '.png')
    plt.close()

plot_backup(wf_bs, 0, 19, 'First_20_Days_Data')
plot_backup(wf_bs, 0, 104, 'First_105_Days_Data')




