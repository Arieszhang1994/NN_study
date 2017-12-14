import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

rng = np.random.RandomState(42)

def build_vocab(file_path):
    vocab = set()
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        vocab.update(words)

    w2i = {w: np.int32(i+2) for i, w in enumerate(vocab)}
    w2i['<s>'], w2i['</s>'] = np.int32(0), np.int32(1) # 文の先頭・終端記号

    return w2i

def encode(sentence, w2i):
    encoded_sentence = []
    for w in sentence:
        encoded_sentence.append(w2i[w])
    return encoded_sentence

def load_data(file_path, vocab=None, w2i=None):
    if vocab is None and w2i is None:
        w2i = build_vocab(file_path)
    
    data = []
    for line in open(file_path, encoding='utf-8'):
        s = line.strip().split()
        s = ['<s>'] + s + ['</s>']
        enc = encode(s, w2i)
        data.append(enc)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w

# 英語->日本語
train_X, e_w2i, e_i2w = load_data('train.en')
train_y, j_w2i, j_i2w = load_data('train.ja')

train_X, _, train_y, _ = train_test_split(train_X, train_y, test_size=0.5, random_state=42) # 演習用に縮小
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.02, random_state=42)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.02, random_state=42)

class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(rng.randn(vocab_size, emb_dim).astype('float32') * scale, name='V')

    def f_prop(self, x):
        return tf.nn.embedding_lookup(self.V, x)
    
    def f_prop_test(self, x_t):
        return tf.nn.embedding_lookup(self.V, x_t)

class LSTM:
    def __init__(self, in_dim, hid_dim, m, h_0=None, c_0=None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # input gate
        self.W_xi = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xi')
        self.W_hi = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hi')
        self.b_i  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_i')
        
        # forget gate
        self.W_xf = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf')
        self.W_hf = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf')
        self.b_f  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_f')

        # output gate
        self.W_xo = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xo')
        self.W_ho = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_ho')
        self.b_o  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_o')

        # cell state
        self.W_xc = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xc')
        self.W_hc = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hc')
        self.b_c  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_c')

        # initial state
        self.h_0 = h_0
        self.c_0 = c_0

        # mask
        self.m = m

    def f_prop(self, x):
        def fn(tm1, x_and_m):
            h_tm1 = tm1[0]
            c_tm1 = tm1[1]
            x_t = x_and_m[0]
            m_t = x_and_m[1]
            # input gate
            i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(h_tm1, self.W_hi) + self.b_i)

            # forget gate
            f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(h_tm1, self.W_hf) + self.b_f)

            # output gate
            o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(h_tm1, self.W_ho) + self.b_o)

            # cell state
            c_t = f_t * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(h_tm1, self.W_hc) + self.b_c)
            c_t = m_t[:, np.newaxis] * c_t + (1. - m_t[:, np.newaxis]) * c_tm1 # Mask

            # hidden state
            h_t = o_t * tf.nn.tanh(c_t)
            h_t = m_t[:, np.newaxis] * h_t + (1. - m_t[:, np.newaxis]) * h_tm1 # Mask

            return [h_t, c_t]

        _x = tf.transpose(x, perm=[1, 0, 2])
        _m = tf.transpose(self.m)

        if self.h_0 == None:
            self.h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))
        if self.c_0 == None:
            self.c_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))

        h, c = tf.scan(fn=fn, elems=[_x, _m], initializer=[self.h_0, self.c_0])
        return tf.transpose(h, perm=[1, 0, 2]), tf.transpose(c, perm=[1, 0, 2])
    
    def f_prop_test(self, x_t):
        # input gate
        i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(self.h_0, self.W_hi) + self.b_i)

        # forget gate
        f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(self.h_0, self.W_hf) + self.b_f)

        # output gate
        o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(self.h_0, self.W_ho) + self.b_o)

        # cell state
        c_t = f_t * self.c_0 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(self.h_0, self.W_hc) + self.b_c)

        # hidden state
        h_t = o_t * tf.nn.tanh(c_t)

        return [h_t, c_t]

class BiLSTM:
    def __init__(self, in_dim, hid_dim, m, h_0=None, c_0=None, h_0_b=None, c_0_b=None):
        hid_dim = int(hid_dim/2)
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # input gate
        self.W_xi = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xi')
        self.W_hi = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hi')
        self.b_i  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_i')
        
        # forget gate
        self.W_xf = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf')
        self.W_hf = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf')
        self.b_f  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_f')

        # output gate
        self.W_xo = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xo')
        self.W_ho = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_ho')
        self.b_o  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_o')

        # cell state
        self.W_xc = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xc')
        self.W_hc = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hc')
        self.b_c  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_c')

        # initial state
        self.h_0 = h_0
        self.c_0 = c_0

        # input gate back
        self.W_xi_b = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xi_b')
        self.W_hi_b = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hi_b')
        self.b_i_b  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_i_b')
        
        # forget gate
        self.W_xf_b = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf_b')
        self.W_hf_b = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf_b')
        self.b_f_b  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_f_b')

        # output gate
        self.W_xo_b = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xo_b')
        self.W_ho_b = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_ho_b')
        self.b_o_b  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_o_b')

        # cell state
        self.W_xc_b = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xc_b')
        self.W_hc_b = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hc_b')
        self.b_c_b  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_c_b')

        # initial state
        self.h_0_b = h_0_b
        self.c_0_b = c_0_b

        # mask
        self.m = m

    def f_prop(self, x):
        def fn(tm1, x_and_m):
            h_tm1 = tm1[0]
            c_tm1 = tm1[1]
            x_t = x_and_m[0]
            m_t = x_and_m[1]
            # input gate
            i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(h_tm1, self.W_hi) + self.b_i)

            # forget gate
            f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(h_tm1, self.W_hf) + self.b_f)

            # output gate
            o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(h_tm1, self.W_ho) + self.b_o)

            # cell state
            c_t = f_t * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(h_tm1, self.W_hc) + self.b_c)
            c_t = m_t[:, np.newaxis] * c_t + (1. - m_t[:, np.newaxis]) * c_tm1 # Mask

            # hidden state
            h_t = o_t * tf.nn.tanh(c_t)
            h_t = m_t[:, np.newaxis] * h_t + (1. - m_t[:, np.newaxis]) * h_tm1 # Mask

            return [h_t, c_t]

        _x = tf.transpose(x, perm=[1, 0, 2])
        _m = tf.transpose(self.m)

        if self.h_0 == None:
            self.h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))
        if self.c_0 == None:
            self.c_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))

        h, c = tf.scan(fn=fn, elems=[_x, _m], initializer=[self.h_0, self.c_0])

        def fn_b(tm1, x_and_m):
            h_tm1 = tm1[0]
            c_tm1 = tm1[1]
            x_t = x_and_m[0]
            m_t = x_and_m[1]
            # input gate
            i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi_b) + tf.matmul(h_tm1, self.W_hi_b) + self.b_i_b)

            # forget gate
            f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf_b) + tf.matmul(h_tm1, self.W_hf_b) + self.b_f_b)

            # output gate
            o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo_b) + tf.matmul(h_tm1, self.W_ho_b) + self.b_o_b)

            # cell state
            c_t = f_t * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc_b) + tf.matmul(h_tm1, self.W_hc_b) + self.b_c_b)
            c_t = m_t[:, np.newaxis] * c_t + (1. - m_t[:, np.newaxis]) * c_tm1 # Mask

            # hidden state
            h_t = o_t * tf.nn.tanh(c_t)
            h_t = m_t[:, np.newaxis] * h_t + (1. - m_t[:, np.newaxis]) * h_tm1 # Mask

            return [h_t, c_t]
        
        _x_b = tf.transpose(x[:,::-1,:], perm=[1, 0, 2])
        _m_b = tf.transpose(self.m[:,::-1])

        if self.h_0_b == None:
            self.h_0_b = tf.matmul(x[:, -1, :], tf.zeros([self.in_dim, self.hid_dim]))
        if self.c_0_b == None:
            self.c_0_b = tf.matmul(x[:, -1, :], tf.zeros([self.in_dim, self.hid_dim]))
        h_b, c_b = tf.scan(fn=fn_b, elems=[_x_b, _m_b], initializer=[self.h_0_b, self.c_0_b])

        return tf.concat([tf.transpose(h, perm=[1, 0, 2]),tf.transpose(h_b, perm=[1, 0, 2])],2), tf.concat([tf.transpose(c, perm=[1, 0, 2]),tf.transpose(c_b, perm=[1, 0, 2])],2)


class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(in_dim + out_dim)),
                        high=np.sqrt(6/(in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(tf.zeros([out_dim], dtype=tf.float32), name='b')
        self.function = function

    def f_prop(self, x):
        return self.function(tf.einsum('ijk,kl->ijl', x, self.W) + self.b)

    def f_prop_test(self, x_t):
        return self.function(tf.matmul(x_t, self.W) + self.b)

e_vocab_size = len(e_w2i)
j_vocab_size = len(j_w2i)
emb_dim = 256
hid_dim = 256*2

x = tf.placeholder(tf.int32, [None, None], name='x')
m = tf.cast(tf.not_equal(x, -1), tf.float32)
d = tf.placeholder(tf.int32, [None, None], name='d')
d_in = d[:, :-1]

d_out = d[:, 1:]
d_out_one_hot = tf.one_hot(d_out, depth=j_vocab_size, dtype=tf.float32)

def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x

encoder = [
    Embedding(e_vocab_size, emb_dim),
    BiLSTM(emb_dim, hid_dim, m)
]

h_enc, c_enc = f_props(encoder, x)

decoder_pre = [
    Embedding(j_vocab_size, emb_dim),
    LSTM(emb_dim, hid_dim, tf.ones_like(d_in, dtype='float32'), h_0=h_enc[:, -1, :], c_0=c_enc[:, -1, :]),
]

decoder_post = [
    Dense(hid_dim, j_vocab_size, tf.nn.softmax)
]

h_dec, c_dec = f_props(decoder_pre, d_in)
y = f_props(decoder_post, h_dec)

cost = -tf.reduce_mean(tf.reduce_sum(d_out_one_hot * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=[1, 2]))

train = tf.train.AdamOptimizer().minimize(cost)

train_X_lens = [len(com) for com in train_X]
sorted_train_indexes = sorted(range(len(train_X_lens)), key=lambda x: -train_X_lens[x])

train_X = [train_X[ind] for ind in sorted_train_indexes]
train_y = [train_y[ind] for ind in sorted_train_indexes]

n_epochs = 10
batch_size = 128
n_batches = len(train_X) // batch_size

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    # train
    train_costs = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        train_X_mb = np.array(pad_sequences(train_X[start:end], padding='post', value=-1))
        train_y_mb = np.array(pad_sequences(train_y[start:end], padding='post', value=-1))

        _, train_cost = sess.run([train, cost], feed_dict={x: train_X_mb, d: train_y_mb})
        train_costs.append(train_cost)

    # valid
    valid_X_mb = np.array(pad_sequences(valid_X, padding='post', value=-1))
    valid_y_mb = np.array(pad_sequences(valid_y, padding='post', value=-1))

    valid_cost = sess.run(cost, feed_dict={x: valid_X_mb, d: valid_y_mb})
    print('EPOCH: %i, Training cost: %.3f, Validation cost: %.3f' % (epoch+1, np.mean(train_cost), valid_cost))


t_0 = tf.constant(0)
y_0 = tf.placeholder(tf.int32, [None, None], name='y_0')
h_0 = tf.placeholder(tf.float32, [None, None], name='h_0')
c_0 = tf.placeholder(tf.float32, [None, None], name='c_0')
f_0 = tf.cast(tf.zeros_like(y_0[:, 0]), dtype=tf.bool) # バッチ内の各サンプルに対して</s>が出たかどうかのflag
f_0_size = tf.reduce_sum(tf.ones_like(f_0, dtype=tf.int32))
max_len = tf.placeholder(tf.int32, name='max_len') # iterationの繰り返し回数の限度

def f_props_test(layers, x_t):
    for layer in layers:
        x_t = layer.f_prop_test(x_t)
    return x_t

def cond(t, h_t, c_t, y_t, f_t):
    num_true = tf.reduce_sum(tf.cast(f_t, tf.int32)) # Trueの数
    unfinished = tf.not_equal(num_true, f_0_size)
    return tf.logical_and(t+1 < max_len, unfinished)

def body(t, h_tm1, c_tm1, y, f_tm1):
    y_tm1 = y[:, -1]

    decoder_pre[1].h_0 = h_tm1
    decoder_pre[1].c_0 = c_tm1
    h_t, c_t = f_props_test(decoder_pre, y_tm1)
    y_t = tf.cast(tf.argmax(f_props_test(decoder_post, h_t), axis=1), tf.int32)

    y = tf.concat([y, y_t[:, np.newaxis]], axis=1)

    f_t = tf.logical_or(f_tm1, tf.equal(y_t, 1)) # flagの更新

    return [t+1, h_t, c_t, y, f_t]

res = tf.while_loop(
    cond,
    body,
    loop_vars=[t_0, h_0, c_0, y_0, f_0],
    shape_invariants=[
        t_0.get_shape(),
        tf.TensorShape([None, None]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None])
    ]
)

valid_X_mb = pad_sequences(valid_X, padding='post', value=-1)
_y_0 = np.zeros_like(valid_X, dtype='int32')[:, np.newaxis]
_h_enc, _c_enc = sess.run([h_enc, c_enc], feed_dict={x: valid_X_mb})
_h_0 = _h_enc[:, -1, :]
_c_0 = _c_enc[:, -1, :]

_, _, _, pred_y, _ = sess.run(res, feed_dict={
    y_0: _y_0,
    h_0: _h_0,
    c_0: _c_0,
    max_len: 100
})

num = 0

origy = valid_X[num][1:-1]
predy = list(pred_y[num])
truey = valid_y[num][1:-1]

print('元の文:', ' '.join([e_i2w[com] for com in origy]))
print('生成文された文:', ' '.join([j_i2w[com] for com in predy[1:predy.index(1)]]))
print('正解文:', ' '.join([j_i2w[com] for com in truey]))