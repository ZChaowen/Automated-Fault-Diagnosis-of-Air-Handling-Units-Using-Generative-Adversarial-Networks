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
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

select_number = 5
size = 30
GAN = True

test_data = np.loadtxt('GAN_data' + str(select_number) + '-1.txt')
# test_data = np.loadtxt('GAN_data5.txt')
test_labels = np.loadtxt('labels1500.txt')

x_test = np.loadtxt('test_data.txt')
y_test = np.loadtxt('test_labels.txt')

min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)

def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


# 加载数据
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
    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)

    train_labels = train_labels.ravel()
    return train_data, train_labels


def training(train_sample, train_label, test_sample):
    # SVM
    train_label = train_label.ravel()
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # 随机森林
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=20)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # 多层感知机
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    # # 决策树
    # dtc = DecisionTreeClassifier()
    # dtc.fit(train_sample, train_label)
    # dt_predict = dtc.predict(test_sample)
    return SVM_predict, RF_predict, MLP_predict


# Vote
def VOTE(test_sample, test_label, SVM_pre, RF_pre, dt_pre):
    test_label = test_label.reshape(-1, 1)
    Votenum = np.array([])
    labels = np.array([])
    delet1 = np.array([]).astype('int32')

    for i in range(test_label.shape[0]):
        Vote = 0
        lab = 0
        if (SVM_pre[i] == RF_pre[i]) & (SVM_pre[i] == dt_pre[i]):
            Vote = 3
            lab = SVM_pre[i]
        if (SVM_pre[i] == RF_pre[i]) & (SVM_pre[i] != dt_pre[i]):
            Vote = 2
            lab = SVM_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (SVM_pre[i] == dt_pre[i]):
            Vote = 2
            lab = SVM_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (RF_pre[i] == dt_pre[i]):
            Vote = 2
            lab = RF_pre[i]
        if (SVM_pre[i] != RF_pre[i]) & (SVM_pre[i] != dt_pre[i]) & (dt_pre[i] != RF_pre[i]):
            Vote = 0
            lab = RF_pre[i]

        labels = np.append(labels, lab)
        Votenum = np.append(Votenum, Vote)
    for j in range(test_label.shape[0]):
        if Votenum[j] < 2:
            delet1 = np.append(delet1, j).astype('int32')
            delet1 = delet1.astype('int32')

    x_train = np.delete(test_sample, delet1, axis=0)
    y_train = np.delete(test_label, delet1, axis=0)
    del_label = test_label[delet1, :]
    del_sample = test_sample[delet1, :]
    labels = np.delete(labels, delet1, axis=0)
    lens = labels.shape[0]
    return x_train, y_train, labels, lens, del_label, del_sample


def CHOSE():
    delet2 = np.array([])
    train_sample, train_label = LOAD_DATA(select_number, size)
    SVM_pre, RF_pre, dt_pre = training(train_sample, train_label, test_data)
    x_train, y_train, label, lens, del_label, del_sample = VOTE(test_data, test_labels, SVM_pre, RF_pre, dt_pre)

    for i in range(lens):
        if label[i] != y_train[i]:
            delet2 = np.append(delet2, i).astype('int32')
            delet2 = delet2.astype('int32')

    chose_data = np.delete(x_train, delet2, axis=0)
    chose_labels = np.delete(y_train, delet2, axis=0)

    train_sample = np.row_stack((train_sample, chose_data))
    train_label = train_label.reshape(-1, 1)
    chose_labels = chose_labels.reshape(-1, 1)
    train_label = np.row_stack((train_label, chose_labels))
    x_train1 = chose_data
    y_train1 = chose_labels

    while y_train1.shape[0] <= 8000:
        del_label = np.append(del_label, y_train[delet2, :])
        del_sample = np.row_stack((del_sample, x_train[delet2, :]))
        SVM_pre, RF_pre, dt_pre = training(train_sample, train_label, test_data)
        x_train, y_train, label, lens, del_label, del_sample = VOTE(del_sample, del_label, SVM_pre, RF_pre, dt_pre)
        delet2 = np.array([])
        for i in range(lens):
            if label[i] != y_train[i]:
                delet2 = np.append(delet2, i).astype('int32')
                delet2 = delet2.astype('int32')

        chose_data = np.delete(x_train, delet2, axis=0)
        chose_labels = np.delete(y_train, delet2, axis=0)

        train_sample = np.row_stack((train_sample, chose_data))
        train_label = train_label.reshape(-1, 1)
        chose_labels = chose_labels.reshape(-1, 1)
        train_label = np.row_stack((train_label, chose_labels))

        x_train1 = np.row_stack((x_train1, chose_data))
        y_train1 = np.append(y_train1, chose_labels)

        if chose_labels.shape[0] <= 10:
            break
    return x_train1, y_train1


def classify_1(x_train1, y_train1):

    sample,label = LOAD_DATA(select_number, size)
    # x_train1 = np.row_stack((x_train1, sample))
    # y_train1 = np.append(y_train1, label)
    y_train1 = y_train1.ravel()
    # 随机森林
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=20)
    y_train1 = y_train1.reshape(y_train1.shape[0], )
    rfc1.fit(x_train1, y_train1)
    RF_pre = rfc1.predict(x_test)
    RF_AC = accuracy_score(y_test, RF_pre)
    f1_rf = f1_score(y_test, RF_pre, average='macro')
    #     print("随机森林1基准测试集验证得分:"+str(accuracy_score(y_test, RF_pre)))
    #     print("=============")

    # SVM
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(x_train1, y_train1)
    clf.predict(x_train1)
    test_pre = clf.predict(x_test)
    SVM_AC = accuracy_score(y_test, test_pre)
    f1_SVM = f1_score(y_test, test_pre, average='macro')
    #     print("SVM基准测试集验证得分:"+str(accuracy_score(y_test, test_pre)))

    # 决策树
    #     print("=============")
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train1, y_train1)
    dt_pre = dtc.predict(x_test)
    DT_AC = accuracy_score(y_test, dt_pre)
    f1_dt = f1_score(y_test, dt_pre, average='macro')
    #     print("决策树基准测试集验证得分:"+str(accuracy_score(y_test, dt_pre)))

    # 贝叶斯
    #     print("=============")
    # mnb = MultinomialNB()
    # mnb.fit(x_train1, y_train1)
    # NB_predict = mnb.predict(x_test)
    # NB_AC = accuracy_score(y_test, NB_predict)
    # print("贝叶斯网络测试集验证得分:" + str(accuracy_score(y_test, NB_predict)))

    # 多层感知机
    #     print("=============")
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(x_train1, y_train1)
    MLP_predict = MLP.predict(x_test)
    MLP_AC = accuracy_score(y_test, MLP_predict)
    f1_MLP = f1_score(y_test, MLP_predict, average='macro')
    #     print("MPL测试集验证得分:"+str(accuracy_score(y_test, MLP_predict)))

    # KNN
    #     print("=============")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train1, y_train1)
    knn_predict = knn.predict(x_test)
    KNN_AC = accuracy_score(y_test, knn_predict)
    f1_KNN = f1_score(y_test, knn_predict, average='macro')
    #     print("KNN测试集验证得分:"+str(accuracy_score(y_test, knn_predict)))

    # LogisticRegression
    # print("===== ensemble evaluation... =======")
    # classifier = LogisticRegression()
    # classifier.fit(x_train1, y_train1)
    # lg_predict = classifier.predict(x_test)
    # LG_AC = accuracy_score(y_test, lg_predict)
    # print("逻辑回归测试集验证得分:" + str(accuracy_score(y_test, lg_predict)))
    return RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, f1_rf, f1_SVM, f1_KNN, f1_MLP, f1_dt

if GAN == True:
    x_train, y_train = CHOSE()
else:
    x_train, y_train = LOAD_DATA(select_number, size)

for i in tqdm(range(30)):
    RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, f1_rf, f1_SVM,\
    f1_KNN, f1_MLP, f1_dt = classify_1(x_train, y_train)
    if i == 0:
        ave_RF = RF_AC
        ave_SVM = SVM_AC
        ave_DT = DT_AC
        ave_MLP = MLP_AC
        ave_KNN = KNN_AC

        ave_f1_RF = f1_rf
        ave_f1_SVM = f1_SVM
        ave_f1_DT = f1_dt
        ave_f1_MLP = f1_MLP
        ave_f1_KNN = f1_KNN
    else:
        ave_RF = np.append(ave_RF, RF_AC)
        ave_SVM = np.append(ave_SVM, SVM_AC)
        ave_DT = np.append(ave_DT, DT_AC)
        ave_MLP = np.append(ave_MLP, MLP_AC)
        ave_KNN = np.append(ave_KNN, KNN_AC)

        ave_f1_RF = np.append(ave_f1_RF, f1_rf)
        ave_f1_SVM = np.append(ave_f1_SVM, f1_SVM)
        ave_f1_DT = np.append(ave_f1_DT, f1_dt)
        ave_f1_MLP = np.append(ave_f1_MLP, f1_MLP)
        ave_f1_KNN = np.append(ave_f1_KNN, f1_KNN)

print('\n')
print('RF accuracy acc: {:.2f}% '.format(Get_Average(ave_RF)*100.0))
print('SVM accuracy acc: {:.2f}% '.format(Get_Average(ave_SVM)*100.0))
print('MLP accuracy acc: {:.2f}% '.format(Get_Average(ave_MLP)*100.0))
print('knn accuracy acc: {:.2f}% '.format(Get_Average(ave_KNN)*100.0))
print('DT accuracy acc: {:.2f}% '.format(Get_Average(ave_DT)*100.0))

print('\n')
print('RF accuracy acc: {:.4f} '.format(Get_Average(ave_f1_RF)))
print('SVM accuracy acc: {:.4f} '.format(Get_Average(ave_f1_SVM)))
print('MLP accuracy acc: {:.4f} '.format(Get_Average(ave_f1_MLP)))
print('knn accuracy acc: {:.4f} '.format(Get_Average(ave_f1_KNN)))
print('DT accuracy acc: {:.4f} '.format(Get_Average(ave_f1_DT)))