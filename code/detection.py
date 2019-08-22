# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:59:12 2018

@author: sn308
"""
import numpy as np
import scipy.io as sio
import random
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model.logistic import LogisticRegression
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

select_number = 50
size = select_number*6

test_data = np.loadtxt('GAN_data' + str(select_number) + '-4.txt')
test_labels = np.loadtxt('labels1500.txt')

def load_normal_data():
    sample = sio.loadmat('data.mat')
    data = sample['ahu']
    chosen = [0, 9, 16, 17, 19, 20, 21, 29, 30, 31, 33, 133]
    for i in range(12):
        if i == 0:
            temp1 = data[:, chosen[i]]
            sample = temp1
        else:
            temp1 = data[:, chosen[i]]
            sample = np.column_stack((sample, temp1))
    num = random.sample(range(0, 21600), 9800)
    labels = sample[num, 0]
    labels = labels-1
    sample = sample[num,1:]
    min_max_scaler = preprocessing.MinMaxScaler()
    sample = min_max_scaler.fit_transform(sample)
    return sample, labels

def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


# load data
def LOAD_DATA(select_number, size):
    t = select_number
    TR_sample_temp = sio.loadmat('AHU.mat')
    data = TR_sample_temp['AHU']
    chosen = [0, 9, 16, 17, 19, 20, 21, 29, 30, 31, 33, 133]
    for i in range(12):

        if i == 0:
            temp1 = data[:, chosen[i]]
            sample = temp1
        else:
            temp1 = data[:, chosen[i]]
            sample = np.column_stack((sample, temp1))
    for i in range(0, 8640, 1440):
        p = 0
        select_number = select_number + i
        num = random.sample(range(i, 1440 + i), t)
        for j in range(i, i + t):
            a = num[p]
            if j == 0:
                temp2 = sample[a, :]
                train_data = temp2.reshape(1, 12)
            else:
                temp2 = sample[a, :]
                temp2 = temp2.reshape(1, 12)
                train_data = np.row_stack((train_data, temp2))
            p = p + 1

    train_labels = train_data[:, 0].reshape(size, 1)
    train_data = np.delete(train_data, [0], axis=1)
    # Normalized
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)

    train_labels = train_labels.ravel()
    return train_data, train_labels


def training1(test_sample):
    # SVM
    train_sample, train_label = LOAD_DATA(select_number, size)
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # Random forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=20)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # Decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)
    return SVM_predict, RF_predict, dt_predict


def training2(test_sample):
    # SVM
    train_sample, train_label = LOAD_DATA(select_number, size)
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # Random forest
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_sample, train_label)
    knn_predict = knn.predict(test_sample)
    return SVM_predict, RF_predict, knn_predict


def training3(test_sample):
    # SVM
    train_sample, train_label = LOAD_DATA(select_number, size)
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # DT
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)

    # knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_sample, train_label)
    knn_predict = knn.predict(test_sample)
    return SVM_predict, dt_predict, knn_predict


def training4(test_sample):
    # SVM
    train_sample, train_label = LOAD_DATA(select_number, size)
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # DT
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)

    # knn
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    return SVM_predict, dt_predict, MLP_predict


def training5(test_sample):
    # SVM
    train_sample, train_label = LOAD_DATA(select_number, size)
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # RF
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # MLP
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    return SVM_predict, RF_predict, MLP_predict


def training6(test_sample):
    train_sample, train_label = LOAD_DATA(select_number, size)
    # SVM
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)

    # RF
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_sample, train_label)
    knn_predict = knn.predict(test_sample)

    return dt_predict, RF_predict, knn_predict


def training7(test_sample):
    train_sample, train_label = LOAD_DATA(select_number, size)
    # DT
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)

    # RF
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # MLP
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    return dt_predict, RF_predict, MLP_predict


def training8(test_sample):
    train_sample, train_label = LOAD_DATA(select_number, size)
    # DT
    dtc = DecisionTreeClassifier()
    dtc.fit(train_sample, train_label)
    dt_predict = dtc.predict(test_sample)

    # RF
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_sample, train_label)
    knn_predict = knn.predict(test_sample)

    # MLP
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    return dt_predict, knn_predict, MLP_predict


def training9(test_sample):
    train_sample, train_label = LOAD_DATA(select_number, size)
    # knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_sample, train_label)
    knn_predict = knn.predict(test_sample)

    # RF
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # MLP
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    return knn_predict, RF_predict, MLP_predict


# Vote
def VOTE(test_sample, test_label):
    test_label = test_label.reshape(-1, 1)
    #     SVM_pre, RF_pre, knn_pre = training1(test_sample)
    #      SVM_pre, RF_pre, knn_pre = training2(test_sample)
    pre1, pre2, pre3 = training1(test_sample)
    Votenum = np.array([])
    labels = np.array([])
    delet1 = np.array([])

    for i in range(test_label.shape[0]):
        Vote = 0
        lab = 0
        if (pre1[i] == pre2[i]) & (pre1[i] == pre3[i]):
            Vote = 3
            lab = pre1[i]
        if (pre1[i] == pre2[i]) & (pre1[i] != pre3[i]):
            Vote = 2
            lab = pre1[i]
        if (pre1[i] != pre2[i]) & (pre1[i] == pre3[i]):
            Vote = 2
            lab = pre1[i]
        if (pre1[i] != pre2[i]) & (pre2[i] == pre3[i]):
            Vote = 2
            lab = pre2[i]
        if (pre1[i] != pre2[i]) & (pre1[i] != pre3[i]) & (pre3[i] != pre2[i]):
            Vote = 0
            lab = pre2[i]

        labels = np.append(labels, lab)
        Votenum = np.append(Votenum, Vote)
    for j in range(test_label.shape[0]):
        if Votenum[j] < 2:
            delet1 = np.append(delet1, j)
            delet1 = delet1.astype('int64')

    x_train = np.delete(test_sample, delet1, axis=0)
    y_train = np.delete(test_label, delet1, axis=0)
    del_label = test_label[delet1, :]
    del_sample = test_sample[delet1, :]
    labels = np.delete(labels, delet1, axis=0)
    lens = labels.shape[0]
    return x_train, y_train, labels, lens, del_label, del_sample


def CHOSE():
    delet2 = np.array([])
    x_train, y_train, label, lens, del_label, del_sample = VOTE(test_data, test_labels)

    for i in range(lens):
        if label[i] != y_train[i]:
            delet2 = np.append(delet2, i).astype('int32')
            delet2 = delet2.astype('int32')

    x_train1 = np.delete(x_train, delet2, axis=0)
    y_train1 = np.delete(y_train, delet2, axis=0)

    while y_train1.shape[0] <= 8000:
        del_label = np.append(del_label, y_train[delet2, :])
        del_sample = np.row_stack((del_sample, x_train[delet2, :]))
        x_train, y_train, label, lens, del_label, del_sample = VOTE(del_sample, del_label)
        delet2 = np.array([])
        for i in range(lens):
            if label[i] != y_train[i]:
                delet2 = np.append(delet2, i)
                delet2 = delet2.astype('int32')

        x_temp = np.delete(x_train, delet2, axis=0)
        y_temp = np.delete(y_train, delet2, axis=0)

        x_train1 = np.row_stack((x_train1, x_temp))
        y_train1 = np.append(y_train1, y_temp)

        if y_temp.shape[0] == 0:
            break
    return x_train1, y_train1

def classify(X_train1, Y_train1, X_test, Y_test):

     # Random forest
     rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=2)
     Y_train1 = Y_train1.reshape(Y_train1.shape[0], )

     rfc1.fit(X_train1, Y_train1)
     RF_pre = rfc1.predict(X_test)
     RF_AC = accuracy_score(Y_test, RF_pre)
     f1_RF = f1_score(Y_test, RF_pre, average='macro')

     # SVM
     clf = SVC(kernel='rbf', C=75, gamma=8)
     clf.set_params(kernel='rbf', probability=True).fit(X_train1, Y_train1)
     clf.predict(X_train1)
     test_pre = clf.predict(X_test)
     SVM_AC = accuracy_score(Y_test, test_pre)
     f1_SVM = f1_score(Y_test, test_pre, average='macro')

     #  Decision tree
     dtc = DecisionTreeClassifier()
     dtc.fit(X_train1, Y_train1)
     dt_pre = dtc.predict(X_test)
     DT_AC = accuracy_score(Y_test, dt_pre)
     f1_DT = f1_score(Y_test, dt_pre, average='macro')

     # MLP

     MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                         hidden_layer_sizes=(100, 3), random_state=1)
     MLP.fit(X_train1, Y_train1)
     MLP_predict = MLP.predict(X_test)
     MLP_AC = accuracy_score(Y_test, MLP_predict)
     f1_MLP = f1_score(Y_test, MLP_predict, average='macro')

     # KNN
     knn = KNeighborsClassifier(n_neighbors=3)
     knn.fit(X_train1, Y_train1)
     knn_predict = knn.predict(X_test)
     KNN_AC = accuracy_score(Y_test, knn_predict)
     f1_KNN = f1_score(Y_test, knn_predict, average='macro')

     # print("===== evaluation... =======")
     return RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, f1_RF, f1_SVM, f1_DT, f1_MLP, f1_KNN


print("============training=============")

n_sample, n_labels = load_normal_data()
S_train = n_sample[:7999,:]
S_test = n_sample[8000:,:]
L_train = n_labels[:7999]
L_test = n_labels[8000:]

x_train1, y_train1 = CHOSE()
y = y_train1.tolist()
count = Counter(y)
y_train1 = np.ones_like(y_train1)

x_test = np.loadtxt('test_data.txt')
y_test = np.loadtxt('test_labels.txt')
min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
y_test = np.ones_like(y_test)

x_test = np.append(x_test, S_test, axis=0)
y_test = np.append(y_test, L_test)

Train_data = np.append(x_train1, S_train, axis=0)
Train_labels = np.append(y_train1, L_train)
Train_labels = Train_labels.ravel()
sample = np.append(Train_data, Train_labels.reshape(-1, 1), axis=1)
np.random.shuffle(sample)
x_train = sample[ : ,0:-1]
y_train = sample[ : , -1]

for i in tqdm(range(30)):
     # RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC = classify()
     RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC ,f1_RF, f1_SVM, f1_DT, \
     f1_MLP, f1_KNN= classify(x_train, y_train, x_test, y_test)
     if i == 0:
          ave_RF = RF_AC
          ave_SVM = SVM_AC
          ave_DT = DT_AC
          ave_MLP = MLP_AC
          ave_KNN = KNN_AC

          ave_f1_RF = f1_RF
          ave_f1_SVM = f1_SVM
          ave_f1_DT = f1_DT
          ave_f1_MLP = f1_MLP
          ave_f1_KNN = f1_KNN
     else:
          ave_RF = np.append(ave_RF, RF_AC)
          ave_SVM = np.append(ave_SVM, SVM_AC)
          ave_DT = np.append(ave_DT, DT_AC)
          ave_MLP = np.append(ave_MLP, MLP_AC)
          ave_KNN = np.append(ave_KNN, KNN_AC)

          ave_f1_RF = np.append(ave_f1_RF, f1_RF)
          ave_f1_SVM = np.append(ave_f1_SVM, f1_SVM)
          ave_f1_DT = np.append(ave_f1_DT, f1_DT)
          ave_f1_MLP = np.append(ave_f1_MLP, f1_MLP)
          ave_f1_KNN = np.append(ave_f1_KNN, f1_KNN)

print("The number of fault samples for each type is：", count)
print('RF accuracy acc: {:.2f}% '.format(Get_Average(ave_RF)*100.0))
print('SVM accuracy acc: {:.2f}% '.format(Get_Average(ave_SVM)*100.0))
print('MLP accuracy acc: {:.2f}% '.format(Get_Average(ave_MLP)*100.0))
print('knn accuracy acc: {:.2f}% '.format(Get_Average(ave_KNN)*100.0))
print('DT accuracy acc: {:.2f}% '.format(Get_Average(ave_DT)*100.0))

print('RF accuracy acc: {:.4f}'.format(Get_Average(ave_f1_RF)))
print('SVM accuracy acc: {:.4f}'.format(Get_Average(ave_f1_SVM)))
print('MLP accuracy acc: {:.4f}'.format(Get_Average(ave_f1_MLP)))
print('knn accuracy acc: {:.4f}'.format(Get_Average(ave_f1_KNN)))
print('DT accuracy acc: {:.4f}'.format(Get_Average(ave_f1_DT)))


print("=========training=========")
# n_sample, n_labels = load_normal_data()
# S_train = n_sample[:7999,:]
# S_test = n_sample[8000:,:]
# L_train = n_labels[:7999]
# L_test = n_labels[8000:]

x_train1, y_train1 = LOAD_DATA(select_number, size)

x_test = np.loadtxt('test_data.txt')
y_test = np.loadtxt('test_labels.txt')
min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
y_test = np.ones_like(y_test)

x_test = np.append(x_test, S_test, axis=0)
y_test = np.append(y_test, L_test)

Train_data = np.append(x_train1, S_train, axis=0)
Train_labels = np.append(y_train1, L_train)
Train_labels = Train_labels.ravel()
sample = np.append(Train_data, Train_labels.reshape(-1, 1), axis=1)
np.random.shuffle(sample)
x_train = sample[ : ,0:-1]
y_train = sample[ : , -1]

for i in tqdm(range(30)):
     # RF_AC, SVM_AC, DT_AC, NB_AC, MLP_AC, KNN_AC, LG_AC = classify()
     RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC ,f1_RF, f1_SVM, f1_DT, \
     f1_MLP, f1_KNN= classify(x_train, y_train, x_test, y_test)
     if i == 0:
          ave_RF = RF_AC
          ave_SVM = SVM_AC
          ave_DT = DT_AC
          ave_MLP = MLP_AC
          ave_KNN = KNN_AC

          ave_f1_RF = f1_RF
          ave_f1_SVM = f1_SVM
          ave_f1_DT = f1_DT
          ave_f1_MLP = f1_MLP
          ave_f1_KNN = f1_KNN
     else:
          ave_RF = np.append(ave_RF, RF_AC)
          ave_SVM = np.append(ave_SVM, SVM_AC)
          ave_DT = np.append(ave_DT, DT_AC)
          ave_MLP = np.append(ave_MLP, MLP_AC)
          ave_KNN = np.append(ave_KNN, KNN_AC)

          ave_f1_RF = np.append(ave_f1_RF, f1_RF)
          ave_f1_SVM = np.append(ave_f1_SVM, f1_SVM)
          ave_f1_DT = np.append(ave_f1_DT, f1_DT)
          ave_f1_MLP = np.append(ave_f1_MLP, f1_MLP)
          ave_f1_KNN = np.append(ave_f1_KNN, f1_KNN)

print('RF accuracy acc: {:.2f}% '.format(Get_Average(ave_RF)*100.0))
print('SVM accuracy acc: {:.2f}% '.format(Get_Average(ave_SVM)*100.0))
print('MLP accuracy acc: {:.2f}% '.format(Get_Average(ave_MLP)*100.0))
print('knn accuracy acc: {:.2f}% '.format(Get_Average(ave_KNN)*100.0))
print('DT accuracy acc: {:.2f}% '.format(Get_Average(ave_DT)*100.0))

print('RF accuracy acc: {:.4f} '.format(Get_Average(ave_f1_RF)))
print('SVM accuracy acc: {:.4f} '.format(Get_Average(ave_f1_SVM)))
print('MLP accuracy acc: {:.4f} '.format(Get_Average(ave_f1_MLP)))
print('knn accuracy acc: {:.4f} '.format(Get_Average(ave_f1_KNN)))
print('DT accuracy acc: {:.4f} '.format(Get_Average(ave_f1_DT)))