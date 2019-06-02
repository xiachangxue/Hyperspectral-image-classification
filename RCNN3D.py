# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-12-30
# Time: 下午12:42
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import *

import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_data(filename):
    data = []
    data_reader = csv.reader(open(filename, 'r'))
    for row in data_reader:
        data.append(row)
    data = np.array(data, np.float32).reshape([145, 145, 200])
    return data


def load_label(filename):
    label = []
    label_reader = csv.reader(open(filename, 'r'))
    for row in label_reader:
        label.append(row)
    return np.array(label, np.int32).reshape([145, 145])


def preprocess(data):
    data = np.reshape(data, [-1, 200])
    for i in range(200):
        scaler = MinMaxScaler().fit(data[:, i])
        data[:, i] = scaler.transform(data[:, i])
    data = np.reshape(data, [145, 145, 200])
    return data


def gen_batch(data, label):
    all_batch_data = []
    all_batch_label = []
    data = np.pad(data, [[6, 6], [6, 6], [0, 0]], 'constant', constant_values=0)
    for i in range(6, 151):
        for j in range(6, 151):
            if label[i - 6, j - 6] == 0:
                continue
            all_batch_data.append(data[i - 6:i + 7, j - 6:j + 7])
            all_batch_label.append(label[i - 6, j - 6] - 1)
    all_batch_data, all_batch_label = np.array(all_batch_data), np.array(all_batch_label)

    all_batch_data = np.transpose(all_batch_data, (0, 3, 1, 2))
  
    return all_batch_data, all_batch_label

def gen_batch1(data, label):
    all_batch_data1 = []
    all_batch_label = []
    data = np.pad(data, [[3, 3], [3, 3], [0, 0]], 'constant', constant_values=0)
    for i in range(3, 148):
        for j in range(3, 148):
            if label[i - 3, j - 3] == 0:
                continue
            all_batch_data1.append(data[i - 3:i + 4, j - 3:j + 4])
            all_batch_label.append(label[i - 3, j - 3] - 1)
    all_batch_data1, all_batch_label = np.array(all_batch_data1), np.array(all_batch_label)

    all_batch_data1 = np.transpose(all_batch_data1, (0, 3, 1, 2))

    return all_batch_data1, all_batch_label


class Model(object):
    def __init__(self, learning_rate=0.00001):
        self.learning_rate = learning_rate
        self.data = tf.placeholder(tf.float32, shape=[None, 200, 13, 13])
        self.data1 = tf.placeholder(tf.float32, shape=[None, 200, 7, 7])
        self.label = tf.placeholder(tf.int64, shape=[None, ])
        self.build_graph()

    def build_graph(self):
        conv1_weight = self.weight_variable([5, 3, 3, 1, 20])
        conv1_bias = self.bias_variable([20, ])
        conv2_weight = self.weight_variable([5, 3, 3, 20, 20])
        conv2_bias = self.bias_variable([20, ])
        conv3_weight = self.weight_variable([5, 3, 3, 20, 35])
        conv3_bias = self.bias_variable([35, ])
        conv4_weight = self.weight_variable([5, 1, 1, 35, 35])
        conv4_bias = self.bias_variable([35, ])
        conv5_weight = self.weight_variable([5, 1, 1, 35, 35])
        conv5_bias = self.bias_variable([35, ])
        conv6_weight = self.weight_variable([5, 1, 1, 35, 35])
        conv6_bias = self.bias_variable([35, ])  
        conv7_weight = self.weight_variable([5, 1, 1, 35, 35])
        conv7_bias = self.bias_variable([35, ])        

        conv1_weight_w = self.weight_variable([5, 1, 1, 1, 20])
        conv1_bias_w = self.bias_variable([20, ])
        conv2_weight_w = self.weight_variable([5, 1, 1, 20, 20])
        conv2_bias_w = self.bias_variable([20, ])
        conv3_weight_w = self.weight_variable([5, 1, 1, 20, 35])
        conv3_bias_w = self.bias_variable([35, ])
       
        
        conv1_weight_r = self.weight_variable([5, 3, 3, 70, 100])
        conv1_bias_r = self.bias_variable([100, ])
        conv2_weight_r = self.weight_variable([5, 3, 3, 100, 100])
        conv2_bias_r = self.bias_variable([100, ])
        conv3_weight_r = self.weight_variable([5, 3, 3, 100, 150])
        conv3_bias_r = self.bias_variable([150, ])
        conv4_weight_r = self.weight_variable([5, 1, 1, 150, 150])
        conv4_bias_r = self.bias_variable([150, ])
        conv5_weight_r = self.weight_variable([5, 1, 1, 150, 150])
        conv5_bias_r = self.bias_variable([150, ])
        conv6_weight_r = self.weight_variable([5, 1, 1, 150, 150])
        conv6_bias_r = self.bias_variable([150, ])
        conv7_weight_r = self.weight_variable([5, 1, 1,150,150])
        conv7_bias_r = self.bias_variable([150, ])  
        fc_weight1 = self.weight_variable([245, num_class])
        fc_bias1 = self.bias_variable([num_class, ])

        fc_weight = self.weight_variable([150, num_class])
        fc_bias = self.bias_variable([num_class, ])
        l2_reg = tf.constant(0.75, tf.float32, [1, ])

        x = tf.reshape(self.data, shape=[-1, 200, 13, 13, 1])
        y = tf.reshape(self.data1, shape=[-1, 200, 7, 7, 1])

        conv1 = tf.nn.conv3d(y, conv1_weight, [1, 1, 1, 1, 1], padding='VALID') + conv1_bias
        relu1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv3d(relu1, conv2_weight, [1, 2, 1, 1, 1], padding='VALID') + conv2_bias
        relu2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv3d(relu2, conv3_weight, [1, 1, 1, 1, 1], padding='VALID') + conv3_bias
        relu3 = tf.nn.relu(conv3)
        conv4 = tf.nn.conv3d(relu3, conv4_weight, [1, 2, 1, 1, 1], padding='VALID') + conv4_bias
        relu4 = tf.nn.relu(conv4)
        conv5 = tf.nn.conv3d(relu4, conv5_weight, [1, 1, 1, 1, 1], padding='VALID') + conv5_bias
        relu5 = tf.nn.relu(conv5)
        conv6 = tf.nn.conv3d(relu5, conv6_weight, [1, 2, 1, 1, 1], padding='VALID') + conv6_bias
        relu6 = tf.nn.relu(conv6)       
        conv7 = tf.nn.conv3d(relu6, conv7_weight, [1, 2, 1, 1, 1], padding='VALID') + conv7_bias
        relu7 = tf.nn.relu(conv7)    

        conv1_w = tf.nn.conv3d(x, conv1_weight, [1, 1, 1, 1, 1], padding='VALID') + conv1_bias
        relu1_w = tf.nn.relu(conv1_w)
        conv2_w = tf.nn.conv3d(relu1_w, conv2_weight, [1, 2, 1, 1, 1], padding='VALID') + conv2_bias
        relu2_w = tf.nn.relu(conv2_w)
        conv3_w = tf.nn.conv3d(relu2_w, conv3_weight, [1, 1, 1, 1, 1], padding='VALID') + conv3_bias
        relu3_w = tf.nn.relu(conv3_w)

        conv1_w_w = tf.nn.conv3d(y, conv1_weight_w, [1, 1, 1, 1, 1], padding='VALID') + conv1_bias_w
        relu1_w_w = tf.nn.relu(conv1_w_w)
        conv2_w_w = tf.nn.conv3d(relu1_w_w, conv2_weight_w, [1, 2, 1, 1, 1], padding='VALID') + conv2_bias_w
        relu2_w_w = tf.nn.relu(conv2_w_w)
        conv3_w_w = tf.nn.conv3d(relu2_w_w, conv3_weight_w, [1, 1, 1, 1, 1], padding='VALID') + conv3_bias_w
        relu3_w_w = tf.nn.relu(conv3_w_w)

        relu8_w = tf.concat(4, [relu3_w, relu3_w_w])
        conv1_r = tf.nn.conv3d(relu8_w, conv1_weight_r, [1, 1, 1, 1, 1], padding='VALID') + conv1_bias_r
        relu1_r = tf.nn.relu(conv1_r)
        conv2_r = tf.nn.conv3d(relu1_r, conv2_weight_r, [1, 2, 1, 1, 1], padding='VALID') + conv2_bias_r
        relu2_r = tf.nn.relu(conv2_r)
        conv3_r = tf.nn.conv3d(relu2_r, conv3_weight_r, [1, 1, 1, 1, 1], padding='VALID') + conv3_bias_r
        relu3_r = tf.nn.relu(conv3_r)
        conv4_r = tf.nn.conv3d(relu3_r, conv4_weight_r, [1, 2, 1, 1, 1], padding='VALID') + conv4_bias_r
        relu4_r = tf.nn.relu(conv4_r)
        conv5_r = tf.nn.conv3d(relu4_r, conv5_weight_r, [1, 1, 1, 1, 1], padding='VALID') + conv5_bias_r
        relu5_r = tf.nn.relu(conv5_r)
        conv6_r = tf.nn.conv3d(relu5_r, conv6_weight_r, [1, 2, 1, 1, 1], padding='VALID') + conv6_bias_r
        relu6_r = tf.nn.relu(conv6_r)
        conv7_r = tf.nn.conv3d(relu6_r, conv7_weight_r, [1, 1, 1, 1, 1], padding='VALID') + conv7_bias_r
        relu7_r = tf.nn.relu(conv7_r)
     



  


        flatten1 = tf.nn.dropout(tf.reshape(relu7, [-1, 245]), keep_prob=0.75)
        flatten = tf.nn.dropout(tf.reshape(relu7_r, [-1, 150]), keep_prob=0.75)
        self.logits1 = tf.nn.xw_plus_b(flatten1, fc_weight1, fc_bias1)
        self.loss1 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits1, self.label) + l2_reg * tf.nn.l2_loss(fc_weight1))

        self.logits = tf.nn.xw_plus_b(flatten, fc_weight, fc_bias)
        self.loss2 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.label) + l2_reg * tf.nn.l2_loss(fc_weight))
        self.loss = self.loss1+self.loss2
        self.prediction = tf.argmax(self.logits, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape), name='bias')


if __name__ == '__main__':
    data_filename = 'data.csv'
    train_label_filename = 'train_label.csv'
    test_label_filename = 'test_label.csv'
    rcnn3d7_filename = 'rcnn3d7.txt'
    batch_size = 100
    learning_rate = 0.0001
    num_class = 16
    num_epoch = 500
    step = 100
    tf.set_random_seed(123)

    data = load_data(data_filename)
    train_label = load_label(train_label_filename)
    test_label = load_label(test_label_filename)
    data = preprocess(data)
    train_batch_data, train_batch_label = gen_batch(data, train_label)
    train_batch_data1, train_batch_label = gen_batch1(data, train_label)


    val_batch_data, val_batch_label = gen_batch(data, test_label)
    val_batch_data1, val_batch_label = gen_batch1(data, test_label)


    rand_ix = np.random.permutation(len(train_batch_data))
    train_batch_data, train_batch_data1,  train_batch_label = train_batch_data[rand_ix], train_batch_data1[rand_ix], train_batch_label[rand_ix]
    rand_ix1 = np.random.permutation(len(val_batch_data))
    val_batch_data, val_batch_data1, val_batch_label = val_batch_data[rand_ix1], val_batch_data1[rand_ix1],val_batch_label[rand_ix1]
    model = Model(learning_rate)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    rcnn3d7 = []
    with tf.Session() as sess:
        sess.run(init)

        for i_epoch in range(num_epoch):

            # training step
            total_train_loss = 0.
            total_train_acc = 0.
            for i in range(0, len(train_batch_data), batch_size):
                if i + batch_size >= len(train_batch_data):
                    break
                batch_data = train_batch_data[i:i + batch_size]
                batch_data1 = train_batch_data1[i:i + batch_size]

                batch_label = train_batch_label[i:i + batch_size]


                batch_data = np.reshape(batch_data, [-1, 200, 13, 13])
                batch_data1 = np.reshape(batch_data1, [-1, 200, 7, 7])


                _, loss, accuracy = sess.run([model.train_op, model.loss, model.accuracy],
                                             feed_dict={model.data: batch_data, model.data1: batch_data1,  model.label: batch_label})
                total_train_loss += loss
                total_train_acc += accuracy

             #   if i / batch_size % step == 0 and i != 0:
              #      print( 'train: ', total_train_loss / step, total_train_acc / step)
               #     total_train_loss = 0.
                #    total_train_acc = 0.

            # validation step
            total_val_loss = 0.
            total_val_acc = 0.
            for i in range(0, len(val_batch_data), batch_size):
                if i + batch_size >= len(val_batch_data):
                    break
                batch_data = val_batch_data[i:i + batch_size]
                batch_data1 = val_batch_data1[i:i + batch_size]

                batch_label = val_batch_label[i:i + batch_size]

                loss, accuracy = sess.run([model.loss, model.accuracy],
                                          feed_dict={model.data: batch_data, model.data1: batch_data1,
                                                     model.label: batch_label})
                total_val_loss += loss
                total_val_acc += accuracy

            cnt = len(val_batch_data) / batch_size
            print ('Epoch ', i_epoch, 'val: ', total_val_loss / cnt, total_val_acc / cnt)
            rcnn3d7.append(total_val_acc / cnt)
            if (i_epoch+1) % 200 == 0:
                with open(rcnn3d7_filename, 'w') as f:
                    for x in rcnn3d7:
                        print (x,file=f)


