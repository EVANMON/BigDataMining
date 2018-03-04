import numpy as np
import csv
import matplotlib.pyplot as plt

# a class that stores ratings for one user
class user_rating:
    def __init__(self, userid, movies, ratings):
        self.userid = userid # a int number
        self.movies = movies # a string list
        self.avg = np.mean(ratings) # a float number
        self.ratings = dict(zip(movies, ratings)) # a dictionary mapping ratings to movies
        G = set() # movies the user likes
        for key, val in self.ratings.items():
            if val >= 3:
                G.add(key)
        self.G = G
                
    def display(self):
        print (self.userid)
        print (self.ratings)
        print (self.G)

# calculate Pearson-correlation coefficient      
def pearson(u, v):
    I = list(set(u.movies)&set(v.movies))
    if not I:
        return 0
    else:
        uk = []
        vk = []
        for id in I:
            uk.append(u.ratings[id])
            vk.append(v.ratings[id])
        uk = np.asarray(uk)
        vk = np.asarray(vk)
        uk -= u.avg
        vk -= v.avg
        tmp1 = np.sum(uk*vk)
        tmp2 = np.sqrt(np.sum(np.square(uk)))
        tmp3 = np.sqrt(np.sum(np.square(vk)))
        p = tmp1/(tmp2*tmp3+1e-9) # to avoid division by zero
        return p

# predict movie ratings of users in testset using information from trainset        
def knn(trainset, testset, k):
    res = []
    for user_test in testset:
        p = []
        for user_train in trainset:
            p.append(pearson(user_test, user_train))
        idx = np.argsort(p)[::-1][:k]
        topk = []
        for i in range(k):
            topk.append(trainset[idx[i]])
        sum = 0
        for i in idx:
            sum += abs(p[i])
        pred = []
        for movie in user_test.movies:
            tmp = user_test.avg
            for user in topk:
                if movie in user.movies:
                    tmp += pearson(user_test, user)*(user.ratings[movie]-user.avg)/sum
            pred.append(tmp)
        res.append(dict(zip(user_test.movies, pred)))
    return res

# give top t movies
def rank(pred, t):
    movies = list(pred.keys())
    ratings = list(pred.values())
    idx = np.argsort(ratings)[::-1][:t]
    recommend = []
    for i in range(t):
        recommend.append(movies[idx[i]])
    return recommend

# cross validation to get precision and recall given a spceific t
def rank_t(users, t):
    num_folds = 10
    folds = []
    precision_folds = []
    recall_folds = []
    for i in range(num_folds):
        folds.append(users[int(i*len(users)/num_folds):int((i+1)*len(users)/num_folds)])
    for i in range(num_folds):
        testset = folds[i]
        trainset = []
        for j in range(num_folds):
            if j != i:
                trainset.extend(folds[j])
        res = knn(trainset, testset, k = 30)
        precision = 0
        recall = 0
        size = len(testset)
        for k in range(size):        
            G = testset[k].G
            if len(testset[k].movies) < t or not G:
                size -= 1
                continue
            else:
                S = set(rank(res[k], t))
                I = S&G
                precision += len(I)/len(S)
                recall += len(I)/len(G)
        if size:
            precision_folds.append(precision/size)
            recall_folds.append(recall/size)
        else:
            continue
    precision_t = np.mean(precision_folds)
    recall_t = np.mean(recall_folds)
    print ('t = ' + str(t) + ' is done!')
    return precision_t, recall_t

# create a list that contains 671 user_rating objects
data = []    
with open('ratings.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] != 'userId':
            data.append((int(row[0]), row[1], float(row[2])))
data.append((0,0,0)) # a dummy data point to avoid index out of bounds
users = []
pointer = 0
for id in range(1, 672):
    movies = []
    ratings = []
    flag = 1
    while flag:
        if data[pointer][0] == id:
            movies.append(data[pointer][1])
            ratings.append(data[pointer][2])
        if data[pointer+1][0] == id:
            pointer += 1
        else:
            flag = 0
    users.append(user_rating(id, movies, ratings))

# test on different t
ts = list(range(1, 26))
precision_knn = []
recall_knn = []
for t in ts:
    p, r = rank_t(users, t)
    precision_knn.append(p)
    recall_knn.append(r)
plt.plot(ts, precision_knn, ts, recall_knn)
plt.xlabel('t')
plt.ylabel('precision and recall')
plt.title('t vs precision and recall using knn')
plt.legend(['precision', 'recall'])
plt.show()
plt.plot(recall_knn, precision_knn)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('recall vs precision using knn')
plt.show()    