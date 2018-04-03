# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:39:36 2018

@author: roy
"""

import numpy as np, pandas as pd, re, matplotlib.pyplot as plt
from datetime import *; from dateutil.parser import parser
data = pd.read_csv('C:/Users/roy/Documents/z_dataset/dataset_run.csv')

def trans(x, data):
    y = re.split(':', data.iloc[x, 1])
    y[3] = str(round(int("{:0<9d}".format(int(y[3])))/1000))
    y = ':'.join(y)
    return y

time = [trans(i, data) for i in range(np.shape(data)[0])]

date_time = [datetime.strptime(data.iloc[i, 0] + ' ' + time[i], '%Y-%m-%d %H:%M:%S:%f') for i in range(np.shape(data)[0])]

data.insert(loc = 0, column = 'datetime', value = date_time)
data = data.drop(columns=['date', 'time', 'username'])
data['datetime'] = pd.to_datetime(data.datetime)
data = data.sort_values(by = 'datetime')
sec_diff = [(data.iloc[i+1, 0] - data.iloc[i, 0]).total_seconds() for i in range(np.shape(data)[0]-1)]
sec_diff.insert(0, 0)

def sec(x) :
    if x>300:
        x = 0
    else:
        x = x
    return x

sec_diff = [sec(sec_diff[i]) for i in range(np.shape(data)[0])]
data.insert(loc = 1, column = 'second diff', value = sec_diff)
ratio = .7; l = np.shape(data)
def analysis(ratio, l, data):
    size = round(ratio*l[0])
    i1 = np.random.choice(range(l[0]), size = size, replace = False)
    train = data.iloc[i1,:]; test = data.drop(data.index[i1])
    train_y = train['activity']; test_y = test['activity']
    train_x = train.drop(columns = ['activity', 'datetime'])
    test_x = test.drop(columns = ['activity', 'datetime'])
    
    from sklearn import ensemble, svm, neighbors, metrics, preprocessing
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import BernoulliNB
    
    forest = ensemble.RandomForestClassifier(n_estimators = 300)
    forest.fit(train_x, train_y)
    forest_p_train = forest.predict(train_x); forest_p_test = forest.predict(test_x)
    
    bag = ensemble.BaggingClassifier(n_estimators = 300)
    bag.fit(train_x, train_y)
    bag_p_train = bag.predict(train_x); bag_p_test = bag.predict(test_x)
    
    boost = ensemble.AdaBoostClassifier(n_estimators = 300)
    boost.fit(train_x, train_y)
    boost_p_train = boost.predict(train_x); boost_p_test = boost.predict(test_x)
    
    svc = svm.SVC()
    svc.fit(train_x, train_y)
    svc_p_train = svc.predict(train_x); svc_p_test = svc.predict(test_x)
    
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_x, train_y)
    knn_p_train = knn.predict(train_x); knn_p_test = knn.predict(test_x)
    
    mnb = BernoulliNB()
    mnb.fit(train_x, train_y)
    mnb_p_train = mnb.predict(train_x); mnb_p_test = mnb.predict(test_x)
    
    eva_train = np.matrix([forest_p_train, bag_p_train, boost_p_train, svc_p_train, knn_p_train, mnb_p_train]).T
    eva_test = np.matrix([forest_p_test, bag_p_test, boost_p_test, svc_p_test, knn_p_test, mnb_p_test]).T
    
    train_merge = np.mean(eva_train, axis = 1).round()
    test_merge = np.mean(eva_test, axis = 1).round()
    
    acc_train = metrics.accuracy_score(train_y, train_merge)
    acc_test = metrics.accuracy_score(test_y, test_merge)
    
    acc = [acc_train, acc_test]
    
    return acc

runs = 30
results = [analysis(ratio, l, data) for i in range(runs)]

plot = np.array(results)

plt.boxplot(plot); plt.show()
