# coding : utf-8
# !/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5

import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
random_state = 42


class Autoencoder:
    def __init__(self, vis_dim, hid_dim, W, function=lambda x: x):
        self.W = W
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
        self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
        self.function = function
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.function(u)

    def decode(self, x):
        y = tf.matmul(x, tf.transpose(self.W)) + self.a
        return self.function(y)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tilde_x = x * noise
        reconst_x = self.f_prop(tilde_x)
        error = -tf.reduce_mean(tf.reduce_sum(x * tf.log(reconst_x) + (1. - x) * tf.log(1. - reconst_x), axis=1))
        return error, reconst_x


class BPSingle:
    def __init__(self, in_dim, out_dim, function):
        self.W = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]

        self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.function(u)

    def pretrain(self, x, noise):
        cost, reconst_x = self.ae.reconst_error(x, noise)
        return cost, reconst_x 

def gcn(x):
    mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    std = np.std(x, axis=(1, 2, 3), keepdims=True)
    return (x - mean)/std
            
class ZCAWhitening:
    """
    Usage: 
    zca = ZCAWhitening()
    zca.fit(gcn(x))
    zca_x = zcatransform(gcn(x))
    """
    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.mean = None
        self.ZCA_matrix = None

    def fit(self, x):
        x = x.reshape(x.shape[0], -1)
        self.mean = np.mean(x, axis=0)
        x -= self.mean
        cov_matrix = np.dot(x.T, x) / x.shape[0]
        A, d, _ = np.linalg.svd(cov_matrix)
        self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.epsilon))), A.T)

    def transform(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x -= self.mean
        x = np.dot(x, self.ZCA_matrix.T)
        return x.reshape(shape)

        
class CNNBatchNorm:
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.gamma = tf.Variable(np.ones(shape, dtype='float32'), name='gamma')
        self.beta = tf.Variable(np.zeros(shape, dtype='float32'), name='beta')
        self.epsilon = epsilon

    def f_prop(self, x):
        if len(x.get_shape()) == 2:
            mean, var = tf.nn.moments(x, axes=0, keepdims=True)
            std = tf.sqrt(var + self.epsilon)
        elif len(x.get_shape()) == 4:
            mean, var = tf.nn.moments(x, axes=(0, 1, 2), keep_dims=True)
            std = tf.sqrt(var + self.epsilon)
        normalized_x = (x - mean) / std
        return self.gamma * normalized_x + self.beta


class CNNConv:
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


class CNNPooling:
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)


class CNNFlatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))


class CNNDense:
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


class CNNActivationSingle:
    def __init__(self, function=lambda x: x):
        self.function = function

    def f_prop(self, x):
        return self.function(x)


class NN:
    def __init__(self, inputshape, outputshape, train_x, train_y, pred_x, layer):
        self.layers = []
        for i in layer:
            self.layers.append(i)
        self.train_x = train_x
        self.train_y = train_y
        self.pred_x = pred_x
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.x = tf.placeholder(tf.float32, inputshape)
        self.t = tf.placeholder(tf.float32, outputshape)

    def pre_train(self, n_epochs, batch_size, learning_rate, corruption_level):
        X = np.copy(self.train_x)
        for l, layer in enumerate(self.layers[:-1]):
            n_batches = self.train_x.shape[0] // batch_size
            x = tf.placeholder(tf.float32)
            noise = tf.placeholder(tf.float32)

            cost, reconst_x = layer.pretrain(x, noise)
            params = layer.params
            train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            encode = layer.f_prop(x)

            for epoch in range(n_epochs):
                X = shuffle(X, random_state=random_state)
                err_all = []
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size

                    _noise = rng.binomial(size=X[start:end].shape, n=1, p=1-corruption_level)
                    _, err = self.sess.run([train, cost], feed_dict={x: X[start: end], noise: _noise})
                    err_all.append(err)
                print('Pretraining:: layer: %d, Epoch: %d, Error: %lf' % (l+1, epoch+1, np.mean(err)))
            X = self.sess.run(encode, feed_dict={x:X})

    def train(self, n_epochs, batch_size, learning_rate):
        # self.pre_train(10,10,0.1,0)
        def f_props(layers, x):
            for layer in layers:
                x = layer.f_prop(x)
            return x

        y = f_props(self.layers, self.x)
        cost = -tf.reduce_mean(tf.reduce_sum(self.t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
        # You can change Optimizers here https://www.tensorflow.org/api_guides/python/train
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        valid = tf.argmax(y, 1)
        n_batches = self.train_x.shape[0] // batch_size
        for epoch in range(n_epochs):
            costsum = 0
            train_x, train_y = shuffle(self.train_x, self.train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                _, _cost = self.sess.run([train, cost], feed_dict={self.x: train_x[start: end], self.t: train_y[start: end]})
                costsum += _cost
            print('EPOCH: %i, cost: %.3f' % (epoch + 1, costsum))  

    def predict(self):
        def f_props(layers, x):
            for layer in layers:
                x = layer.f_prop(x)
            return x

        y = f_props(self.layers, self.x)
        valid = tf.argmax(y, 1)
        pred_y_m = self.sess.run([valid], feed_dict={self.x: self.pred_x})
        pred_y = pred_y_m[0]
        self.sess.close()
        return pred_y


#############################################################################
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X = np.r_[mnist.train.images, mnist.test.images]
    mnist_y = np.r_[mnist.train.labels, mnist.test.labels]
    return train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)


train_X, test_X, train_y, test_y = load_mnist()

# validate for small dataset
train_X_mini = train_X[:1000].reshape(train_X[:1000].shape[0], 28, 28, 1)
train_y_mini = train_y[:1000]
test_X_mini = test_X[:200].reshape(test_X[:200].shape[0], 28, 28, 1)
test_y_mini = test_y[:200]

# print(test_X_mini.shape)
a = NN([None, train_X_mini.shape[1], train_X_mini.shape[2], train_X_mini.shape[3]], [
    None, train_y_mini.shape[1]],
        train_X_mini, train_y_mini, test_X_mini, [
        CNNConv((5, 5, 1, 20), tf.nn.relu),    # 24x24x20 -> 12x12x2
        CNNPooling((1, 2, 2, 1)),              # 24x24x20 -> 12x12x20
        CNNConv((5, 5, 20, 50), tf.nn.relu),   # 12x12x20 ->  8x 8x50
        CNNPooling((1, 2, 2, 1)),              # 8x 8x50 ->  4x 4x50
        CNNFlatten(),
        CNNDense(4*4*50, 10, tf.nn.softmax)])
a.train(100, 10, 0.2)
pred_y = a.predict()
print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))