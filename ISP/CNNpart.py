#!/usr/bin/env python3
# -*- coding: utf_8 -*-
'''
CNN parts
'''
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)

class ResUnitC_B_R:
    '''
    Residual block
    '''
    def __init__(self, filter_shape_1, filter_shape_2, function=tf.nn.relu, strides=[1, 1, 1, 1], padding1='SAME',padding2='SAME', down_sample, different_shape, projection_method='1x1conv'):
        # Xavier Initialization
        fan_in_1 = np.prod(filter_shape_1[:3])
        fan_out_1 = np.prod(filter_shape_1[:2]) * filter_shape_1[3]
        fan_in_2 = np.prod(filter_shape_2[:3])
        fan_out_2 = np.prod(filter_shape_2[:2]) * filter_shape_2[3]
        self.W1 = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in_1 + fan_out_1)),
                        high=np.sqrt(6/(fan_in_1 + fan_out_1)),
                        size=filter_shape_1
                    ).astype('float32'), name='W1')
        self.b1 = tf.Variable(np.zeros((filter_shape_1[3]), dtype='float32'), name='b1')
        self.W2 = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in_2 + fan_out_2)),
                        high=np.sqrt(6/(fan_in_2 + fan_out_2)),
                        size=filter_shape_2
                    ).astype('float32'), name='W2')
        self.b2 = tf.Variable(np.zeros((filter_shape_2[3]), dtype='float32'), name='b2')
        self.W_add = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in_1 + fan_out_1)),
                        high=np.sqrt(6/(fan_in_1 + fan_out_1)),
                        size=filter_shape_1
                    ).astype('float32'), name='W_add')
        self.b_add = tf.Variable(np.zeros((filter_shape_1[3]), dtype='float32'), name='b_add')
        self.function = function
        self.strides = strides
        self.padding1 = padding1
        self.padding2 = padding2
        self.down_sample = down_sample
        self.projection_method = projection_method
        self.different_shape = different_shape

    def f_prop(self, x):

        def batch_normalization_layer(input_layer):
            '''
            Helper function to do batch normalziation
            different from the BatchNormation Class beacause it batch is noknown
            :param input_layer: 4D tensor
            :depth: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
            :return: the 4D tensor after being normalized
            '''
            depth = input_layer.get_shape().as_list()[-1] 
            mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
            gamma = tf.Variable(np.ones(depth, dtype='float32'), name='gamma')
            beta = tf.Variable(np.zeros(depth, dtype='float32'), name='beta')
            bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

            return bn_layer
        
        if self.down_sample:
            l1 = tf.nn.conv2d(x, self.W1, strides=[1, 2, 2, 1], padding=self.padding1) + self.b1
        else:
            l1 = tf.nn.conv2d(x, self.W1, strides=self.strides, padding=self.padding1) + self.b1
        l1_acted = self.function(batch_normalization_laye(l1))
        l2 = tf.nn.conv2d(l1_acted, self.W2, strides=self.strides, padding=self.padding2) + self.b2
        if self.down_sample:
            x_add = tf.nn.conv2d(x, self.W_add, strides=[1, 2, 2, 1], padding=self.padding1) + self.b_add
        elif self.different_shape:
            x_add = tf.nn.conv2d(x, self.W_add, strides=self.strides, padding=self.padding1) + self.b_add
        else:
             x_add = x  
        l_added = batch_normalization_laye(l2) + x_add
        return self.function(l_added)

class ResUnitB_R_C:
    '''
    Residual block
    '''
    def __init__(self, filter_shape, function=tf.nn.relu, strides=[1, 1, 1, 1], padding1='SAME',padding2='SAME', down_sample, different_shape, projection_method='1x1conv'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W1 = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W1')
        self.b1 = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b1')
        self.W2 = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W2')
        self.b2 = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b2')
        self.W_add = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W_add')
        self.b_add = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b_add')
        self.function = function
        self.strides = strides
        self.padding1 = padding1
        self.padding2 = padding2
        self.down_sample = down_sample
        self.projection_method = projection_method
        self.different_shape = different_shape

    def f_prop(self, x):

        def batch_normalization_layer(input_layer):
            '''
            Helper function to do batch normalziation
            different from the BatchNormation Class beacause it batch is noknown
            :param input_layer: 4D tensor
            :depth: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
            :return: the 4D tensor after being normalized
            '''
            depth = input_layer.get_shape().as_list()[-1] 
            mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
            gamma = tf.Variable(np.ones(depth, dtype='float32'), name='gamma')
            beta = tf.Variable(np.zeros(depth, dtype='float32'), name='beta')
            bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

            return bn_layer
        r_x = self.function(batch_normalization_laye(x))
        if self.down_sample:
            l1 = tf.nn.conv2d(r_x, self.W1, strides=[1, 2, 2, 1], padding=self.padding1) + self.b1
        else:
            l1 = tf.nn.conv2d(r_x, self.W1, strides=self.strides, padding=self.padding1) + self.b1
        r_l1 = self.function(batch_normalization_laye(l1))
        l2 = tf.nn.conv2d(r_l1, self.W2, strides=self.strides, padding=self.padding2) + self.b2
        if self.down_sample:
            x_add = tf.nn.conv2d(x, self.W_add, strides=[1, 2, 2, 1], padding=self.padding1) + self.b_add
        elif self.different_shape:
            x_add = tf.nn.conv2d(x, self.W_add, strides=self.strides, padding=self.padding1) + self.b_add
        else:
             x_add = x   
        return l2 + x_add

class Conv:
    '''
    Convolutional layer
    '''
    def __init__(self, filter_shape, function=lambda x: x, strides=[1, 1, 1, 1], padding='VALID'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)

class Pooling:
    '''
    Pooling layer
    '''
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', flag='max'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.flag = flag

    def f_prop(self, x):
        if self.flag == 'max':
            y = tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)
        else:
            y = tf.nn.avg_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return y

class Flatten:
    '''
    Flatten layer
    '''
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

class BatchNorm:
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.gamma = tf.Variable(np.ones(shape, dtype='float32'), name='gamma')
        self.beta = tf.Variable(np.zeros(shape, dtype='float32'), name='beta')
        self.epsilon = epsilon

    def f_prop(self, x):
        if len(x.get_shape()) == 2:
            mean, var = tf.nn.moments(x, axes=0, keep_dims=True)
            std = tf.sqrt(var + self.epsilon)
        elif len(x.get_shape()) == 4:
            mean, var = tf.nn.moments(x, axes=(0, 1, 2), keep_dims=True)
            std = tf.sqrt(var + self.epsilon)
        normalized_x = (x - mean) / std
        return self.gamma * normalized_x + self.beta

class Dropout:
    '''
    Dropout
    '''
    def __init__(self, keep_prob):
        self.keep_prob = tf.placeholder("float")
    
    def f_prop(self,x):
        return tf.nn.dropout(x, self.keep_prob)

class Dense:
    '''
    Dense layer
    '''
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier Initialization
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(in_dim + out_dim)),
                        high=np.sqrt(6/(in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

class Activation:
    def __init__(self, function=lambda x: x):
        self.function = function

    def f_prop(self, x):
        return self.function(x)