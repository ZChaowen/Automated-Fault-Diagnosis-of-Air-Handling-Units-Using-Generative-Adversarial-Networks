# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:54:44 2018

@author: Zhongchaowen
"""

import os, time, pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import scipy.io as sio
import tensorflow.contrib.slim as slim

tf.reset_default_graph()
select_number = 40
# training parameters
M_size = 240
N_size = 11

LabM_size = 240
LabN_size = 6

G_size = 240
Zn_size = 100

lr_g = 0.0001
lr_D = 0.0001

train_epoch = 20000

# onehot = np.eye(20)
# variables : input
with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, N_size))
    y = tf.placeholder(tf.float32, shape=(None, LabN_size))
    z = tf.placeholder(tf.float32, shape=(None, Zn_size))
    gy = tf.placeholder(tf.float32, shape=(None, LabN_size))
    isTrain = tf.placeholder(dtype=tf.bool)


# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)
        #        z = slim.fully_connected(cat1, 256, activation_fn = tf.nn.relu)
        #        z = slim.fully_connected(z, 512, activation_fn = tf.nn.relu)
        #        z = slim.fully_connected(z, 256, activation_fn = tf.nn.relu)
        z = slim.fully_connected(cat1, 128, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 64, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 32, activation_fn=tf.nn.relu)
        z = slim.fully_connected(z, 11, activation_fn=tf.nn.relu)

        return z


# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        cat1 = tf.concat([x, y], 1)

        x = slim.fully_connected(cat1, 64, activation_fn=tf.nn.relu)
        #        x = slim.fully_connected(x, 128, activation_fn = tf.nn.relu)
        #        x = slim.fully_connected(x, 256, activation_fn = tf.nn.relu)
        x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 32, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 1, activation_fn=None)

        return x


def load_data(select_num, size, reuse=False):
    t = select_num
    TR_sample_temp = sio.loadmat('AHU.mat')
    data = TR_sample_temp['AHU']
    # 特征选择
    chosen = [0, 9, 16, 17, 19, 20, 21, 29, 30, 31, 33, 133]
    L = len(chosen)
    for i in range(L):
        if i == 0:
            temp1 = data[:, chosen[i]]
            sample = temp1
        else:
            temp1 = data[:, chosen[i]]
            sample = np.column_stack((sample, temp1))
    # 随机选取样本
    for i in range(0, 8640, 1440):
        p = 0
        select_num = select_num + i
        num = random.sample(range(i, 1440 + i), t)
        for j in range(i, i + t):
            a = num[p]
            if j == 0:
                temp2 = sample[a, :]
                train_data = temp2.reshape(1, L)
            else:
                temp2 = sample[a, :]
                temp2 = temp2.reshape(1, L)
                train_data = np.row_stack((train_data, temp2))
            p = p + 1
    train_labels = train_data[:, 0].reshape(size, 1)
    train_data = np.delete(train_data, [0], axis=1)
    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)

    # name1 = "train_data5" + ".txt"
    # name2 = "train_labels5" + ".txt"
    # np.savetxt(name1, train_data)
    # np.savetxt(name2, train_labels)
    return train_data, train_labels


# def load_data():
#    train_data = np.loadtxt('sample50.txt')
#    train_labels = np.loadtxt('labels50.txt')
#    min_max_scaler = preprocessing.MinMaxScaler()
#    train_data = min_max_scaler.fit_transform(train_data)
#    return train_data, train_labels

def one_hot(y, size, reuse=False):
    label = []
    for i in range(size):
        a = int(y[i]) - 1
        temp = [0, 0, 0, 0, 0, 0]
        temp[a] = 1
        if i == 0:
            label = temp
        else:
            label.extend(temp)
    label = np.array(label).reshape(size, 6)
    return label


def G_labels(select_num, size):
    temp_z_ = np.random.uniform(-1, 1, (select_num, size))
    z_ = temp_z_
    fixed_y_ = np.ones((select_num, 1))
    j = 1
    for i in range(5):
        temp = np.ones((select_num, 1)) + j
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)  # 矩阵拼接[10*100]*10
    y = one_hot(fixed_y_, M_size, reuse=True)
    #    name = "labels" + ".txt"
    #    np.savetxt(name, fixed_y_)
    return y, z_


def Generate_date():
    temp_z_ = np.random.uniform(-1, 1, (1500, 100))
    z_ = temp_z_
    fixed_y_ = np.ones((1500, 1))
    j = 1
    for i in range(5):
        temp = np.ones((1500, 1)) + j
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)
        j = j + 1
        z_ = np.concatenate([z_, temp_z_], 0)  # 矩阵拼接[10*100]*10
    y = one_hot(fixed_y_, 9000, reuse=True)
    name = "labels1500" + ".txt"
    np.savetxt(name, fixed_y_)
    return y, z_


def show_result(epoch_num, reuse=False):
    with tf.variable_scope('show_result', reuse=reuse):
        if epoch_num == 19999:
            G_y, fixed_z_ = Generate_date()
            G = sess.run(G_z, {z: fixed_z_, gy: G_y, isTrain: True})
            G_sample = G
            #            name = str(epoch_num) + ".txt"
            name = "GAN_data40-27" + ".txt"
            np.savetxt(name, G_sample)
            return G_sample


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# networks : generator

G_z = generator(z, gy, isTrain)

# Wgan trick

eps = tf.random_uniform(shape=[G_size, 1], minval=0., maxval=1.)
X_inter = eps * x + (1. - eps) * G_z
grad = tf.gradients(discriminator(X_inter, y, isTrain), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

# networks : discriminator
D_real_logits = discriminator(x, y, isTrain, reuse=True)
D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)
# D_real, D_real_logits = discriminator(x, y, isTrain)
# D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# loss for each network

# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([M_size, 1])))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([G_size, 1])))
# D_loss = - (tf.log(D_real_logits) + tf.log(1 - D_fake_logits))   # d_loss大小为 batch_size * 1
# G_loss = - tf.log(D_fake_logits)

# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([G_size, 1])))


D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + grad_pen
G_loss = -tf.reduce_mean(D_fake_logits)
# trainable variables for each network

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr_D, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(G_loss, var_list=G_vars)

# results save folder
root = 'data_results/'
model = 'data_cGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# open session and initialize all variables
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=gpuConfig)
sess.run(tf.global_variables_initializer())

# training-loop

print('training start!')
start_time = time.time()
x_, y_ = load_data(select_number, M_size, reuse=True)
for epoch in range(train_epoch):
    epoch_start_time = time.time()
    # upadta Discriminator

    #    for i in range(5):
    #     x_, y_ = load_data(select_number, M_size,  reuse = True)
    #    x_, y_ = load_data()
    labels = one_hot(y_, M_size, reuse=True)
    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))

    D_losses, _ = sess.run([D_loss, D_optim], {x: x_, y: labels, z: z_, gy: labels, isTrain: True})

    # updata generator

    z_ = np.random.uniform(-1, 1, (G_size, Zn_size))
    G_y, _ = G_labels(select_number, Zn_size)
    G_losses, _, _ = sess.run([G_loss, G_z, G_optim], {x: x_, y: labels, z: z_, gy: G_y, isTrain: True})

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    G = show_result(epoch)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')