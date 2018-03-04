import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise.model_selection.validation import cross_validate
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.matrix_factorization import SVD

def read_data(filename):
    dir = os.path.dirname('__file__')
    f = os.path.join(dir, '..', 'data', filename)
    df = pd.read_csv(f, delimiter=',')
    return df

ratings = read_data('ratings.csv')
movies = read_data('movies.csv')

'''
R = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating', 
                                fill_value=0, dropna=False)
R.unstack()
R = R.unstack().reset_index(name='rating')
R.rename(columns={'level_0': 'userId', 'level_1': 'movieId'}, inplace=True)
'''

# NMF Filter

def NMF_filter(ratings, dims):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    RMSE = np.empty([len(dims)])
    MAE = np.empty([len(dims)])
    min_RMSE = False
    min_MAE = False
    fac_num_RMSE = 0
    fac_num_MAE = 0

    for k in range(len(dims)):
        nmf = NMF(n_factors=dims[k], biased = False)
        cv = cross_validate(algo=nmf, data=data, measures=['RMSE', 'MAE'],
                            cv=10, verbose=True)
        RMSE[k] = np.mean(cv['test_rmse']) 
        if ((not min_RMSE) or RMSE[k] < min_RMSE):
            min_RMSE = RMSE[k]
            fac_num_RMSE = dims[k]

        MAE[k] = np.mean(cv['test_mae'])
        if ((not min_MAE) or MAE[k] < min_MAE):
            min_MAE = MAE[k]
            fac_num_MAE = dims[k]

    plt.plot(dims, RMSE)
    plt.plot(dims, MAE)
    plt.legend(['RMSE', 'MAE'])
    plt.show()
    print ('Finishing Plotting...')
    print ('For RMSE:')
    print ('\t---Optimal number of latent factors is ', fac_num_RMSE)
    print ('\t---Minumun Average RMSE is ', min_RMSE)
    print ('\nFor MAE:')
    print ('\t---Optimal number of latent factors is ', fac_num_MAE)
    print ('\t---Minumun Average MAE is ', min_MAE)

# Q17 & Q18
dims = xrange(2, 51, 2)
NMF_filter(ratings, dims)

# Trimming Filter
def NMF_trim_filter(ratings, dims, func, mv_dict):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    RMSE = np.empty([len(dims)])
    MAE = np.empty([len(dims)])
    min_RMSE = False
    min_MAE = False
    fac_num_RMSE = 0
    fac_num_MAE = 0
    kf = KFold(n_splits=10, random_state=42)

    for k in range(len(dims)):
        nmf = NMF(n_factors=dims[k], random_state=42)
        test_rmse = np.array([])
        test_mae = np.array([])
        for trainset, testset in kf.split(data):
            nmf.fit(trainset)
            full_data = trainset.build_testset() + testset
            func(mv_dict, testset)
            pred = nmf.test(testset)
            test_rmse = np.append(test_rmse, accuracy.rmse(pred, verbose=False))
            test_mae = np.append(test_mae, accuracy.mae(pred, verbose=False))
        RMSE[k] = np.mean(test_rmse) 
        if ((not min_RMSE) or RMSE[k] < min_RMSE):
            min_RMSE = RMSE[k]
            fac_num_RMSE = dims[k]

        MAE[k] = np.mean(test_mae)
        if ((not min_MAE) or MAE[k] < min_MAE):
            min_MAE = MAE[k]
            fac_num_MAE = dims[k]
        print ('For k = %i :' %dims[k])
        print ('RMSE: ', RMSE[k])
        print ('MAE: ', MAE[k])

    plt.plot(dims, RMSE)
    plt.plot(dims, MAE)
    plt.legend(['RMSE', 'MAE'])
    plt.show()
    print ('Finishing Plotting...')
    print ('For RMSE:')
    print ('\t---Optimal number of latent factors is ', fac_num_RMSE)
    print ('\t---Minumun Average RMSE is ', min_RMSE)
    print ('\nFor MAE:')
    print ('\t---Optimal number of latent factors is ', fac_num_MAE)
    print ('\t---Minumun Average MAE is ', min_MAE)
    
def pop_trim(mv_dict, test):
    test[:] = [ts for ts in test if len(mv_dict[ts[1]]) > 2]

def unpop_trim(mv_dict, test):
    test[:] = [ts for ts in test if len(mv_dict[ts[1]]) <= 2]

def high_var_trim(mv_dict, test):
    test[:] = [ts for ts in test 
               if len(mv_dict[ts[1]]) >= 5 and 
               np.var(mv_dict[ts[1]]) >= 2]
    
def movie_counter(full_data):
    mv_ratings = {}
    for data in full_data:
        if data[1] not in mv_ratings.keys():
            mv_ratings[data[1]] = [data[2]]
        else:
            mv_ratings[data[1]].append(data[2])
    return mv_ratings

# Generate the dictionary of all movies
reader = Reader(rating_scale=(0.0, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
empty_ds, full_ds = train_test_split(data, test_size=1.0, random_state=42)
mv_dict = movie_counter(full_ds)

# Q19
dims = xrange(2, 51, 2)
NMF_trim_filter(ratings, dims, pop_trim, mv_dict)

# Q20
dims = xrange(2, 51, 2)
NMF_trim_filter(ratings, dims, unpop_trim, mv_dict)

# Q21
dims = xrange(2, 51, 2)
NMF_trim_filter(ratings, dims, high_var_trim, mv_dict)

# Q22
def plot_roc(pre, tar):
    fpr, tpr, _ = roc_curve(tar, pre)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f' %roc_auc)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    
def NMF_bin_pre(ratings, ts, nmf_fac, thrd):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=ts)
    algo = NMF(n_factors=nmf_fac, random_state=42)
    algo.fit(trainset)
    pre = algo.test(testset)

    true_rating = np.empty(len(pre))
    pred_rating = np.empty(len(pre))

    for i in range(len(pre)):
        true_rating[i] = pre[i][2]
        pred_rating[i] = pre[i][3]

    bi_rating = np.empty(len(pre))
    one_idx = true_rating >= thrd
    zero_idx = true_rating < thrd
    bi_rating[one_idx] = 1.0
    bi_rating[zero_idx] = 0.0
    
    return bi_rating, pred_rating
    

threshold = np.array([2.5, 3, 3.5, 4])
for td in threshold:
    tar, pre = NMF_bin_pre(ratings, 0.1, 18, td)
    plot_roc(pre, tar)

# Q23
reader = Reader(rating_scale=(0.0, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data = data.build_full_trainset()
nmf = NMF(n_factors=20, random_state=42)
nmf.fit(data)
for i in range(10):
    col = nmf.qi[:,i]
    top_movie = col.argsort()[::-1][:10]
    print ('For the %i th column, the top 10 movie genres are:' %(i+1))
    for j in range(10):
        raw_iid = nmf.trainset.to_raw_iid(top_movie[j])
        gen = movies.loc[movies['movieId']==raw_iid]['genres'].values
        print ('\t--%i :' %(j+1), gen)


# MF With Bias Filter
# Q24 & 25

def MF_bias_filter(ratings, dims):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    RMSE = np.empty([len(dims)])
    MAE = np.empty([len(dims)])
    min_RMSE = False
    min_MAE = False
    fac_num_RMSE = 0
    fac_num_MAE = 0

    for k in range(len(dims)):
        svd = SVD(n_factors=dims[k], random_state=42)
        cv = cross_validate(algo=svd, data=data, measures=['RMSE', 'MAE'],
                            cv=10, verbose=True)
        RMSE[k] = np.mean(cv['test_rmse']) 
        if ((not min_RMSE) or RMSE[k] < min_RMSE):
            min_RMSE = RMSE[k]
            fac_num_RMSE = dims[k]

        MAE[k] = np.mean(cv['test_mae'])
        if ((not min_MAE) or MAE[k] < min_MAE):
            min_MAE = MAE[k]
            fac_num_MAE = dims[k]

    plt.plot(dims, RMSE)
    plt.plot(dims, MAE)
    plt.legend(['RMSE', 'MAE'])
    plt.show()
    print ('Finishing Plotting...')
    print ('For RMSE:')
    print ('\t---Optimal number of latent factors is ', fac_num_RMSE)
    print ('\t---Minumun Average RMSE is ', min_RMSE)
    print ('\nFor MAE:')
    print ('\t---Optimal number of latent factors is ', fac_num_MAE)
    print ('\t---Minumun Average MAE is ', min_MAE)

dims = xrange(2, 51, 2)
MF_bias_filter(ratings, dims)

# Trimming Filter
def MF_trim_filter(ratings, dims, func, mv_dict):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    RMSE = np.empty([len(dims)])
    MAE = np.empty([len(dims)])
    min_RMSE = False
    min_MAE = False
    fac_num_RMSE = 0
    fac_num_MAE = 0
    kf = KFold(n_splits=10, random_state=42)

    for k in range(len(dims)):
        svd = SVD(n_factors=dims[k], random_state=42)
        test_rmse = np.array([])
        test_mae = np.array([])
        for trainset, testset in kf.split(data):
            svd.fit(trainset)
            full_data = trainset.build_testset() + testset
            func(mv_dict, testset)
            pred = svd.test(testset)
            test_rmse = np.append(test_rmse, accuracy.rmse(pred, verbose=False))
            test_mae = np.append(test_mae, accuracy.mae(pred, verbose=False))
        RMSE[k] = np.mean(test_rmse) 
        if ((not min_RMSE) or RMSE[k] < min_RMSE):
            min_RMSE = RMSE[k]
            fac_num_RMSE = dims[k]

        MAE[k] = np.mean(test_mae)
        if ((not min_MAE) or MAE[k] < min_MAE):
            min_MAE = MAE[k]
            fac_num_MAE = dims[k]
        print ('For k = %i :' %dims[k])
        print ('RMSE: ', RMSE[k])
        print ('MAE: ', MAE[k])

    plt.plot(dims, RMSE)
    plt.plot(dims, MAE)
    plt.legend(['RMSE', 'MAE'])
    plt.show()
    print ('Finishing Plotting...')
    print ('For RMSE:')
    print ('\t---Optimal number of latent factors is ', fac_num_RMSE)
    print ('\t---Minumun Average RMSE is ', min_RMSE)
    print ('\nFor MAE:')
    print ('\t---Optimal number of latent factors is ', fac_num_MAE)
    print ('\t---Minumun Average MAE is ', min_MAE)

# Q26
dims = xrange(2, 51, 2)
MF_trim_filter(ratings, dims, pop_trim, mv_dict)

# Q27
dims = xrange(2, 51, 2)
MF_trim_filter(ratings, dims, unpop_trim, mv_dict)

# Q28
dims = xrange(2, 51, 2)
MF_trim_filter(ratings, dims, high_var_trim, mv_dict)

# Q29

def plot_roc(pre, tar):
    fpr, tpr, _ = roc_curve(tar, pre)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f' %roc_auc)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    
def MF_bin_pre(ratings, ts, nmf_fac, thrd):
    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=ts)
    algo = SVD(n_factors=nmf_fac, random_state=42)
    algo.fit(trainset)
    pre = algo.test(testset)

    true_rating = np.empty(len(pre))
    pred_rating = np.empty(len(pre))

    for i in range(len(pre)):
        true_rating[i] = pre[i][2]
        pred_rating[i] = pre[i][3]

    bi_rating = np.empty(len(pre))
    one_idx = true_rating >= thrd
    zero_idx = true_rating < thrd
    bi_rating[one_idx] = 1.0
    bi_rating[zero_idx] = 0.0
    
    return bi_rating, pred_rating
    

threshold = np.array([2.5, 3, 3.5, 4])
for td in threshold:
    tar, pre = MF_bin_pre(ratings, 0.1, 6, td)
    plot_roc(pre, tar)