import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def load_amazon(n_features, filepath):
    """
    Load Amazon Reviews
    """
    mat = loadmat(filepath)
    xx = mat['xx']
    yy = mat['yy']
    offset = mat['offset']  # 这个数据文件的组织方式不熟悉

    x = xx[:n_features, :].toarray().T  # n_samples * n_features
    y = yy.ravel()  # 转一维数组

    return x, y, offset


def shuffle(x, y):
    """Shuffle data"""
    shuffled_id = np.arange(x.shape[0])
    np.random.shuffle(shuffled_id)
    x = x[shuffled_id, :]
    y = y[shuffled_id]
    return x, y


def to_one_hot(a):
    b = np.zeros((len(a), 2))
    b[np.arange(len(a)), a] = 1
    return b


def split_data(s_id, t_id, x, y, offset, n_tr_samples, seed=0):
    np.random.seed(seed)
    x_s_tr = x[offset[s_id, 0]:offset[s_id, 0] + n_tr_samples, :]
    x_t_tr = x[offset[t_id, 0]:offset[t_id, 0] + n_tr_samples, :]
    x_s_tst = x[offset[s_id, 0] + n_tr_samples:offset[s_id + 1, 0], :]
    x_t_tst = x[offset[t_id, 0] + n_tr_samples:offset[t_id + 1, 0], :]
    y_s_tr = y[offset[s_id, 0]:offset[s_id, 0] + n_tr_samples]
    y_t_tr = y[offset[t_id, 0]:offset[t_id, 0] + n_tr_samples]
    y_s_tst = y[offset[s_id, 0] + n_tr_samples:offset[s_id + 1, 0]]
    y_t_tst = y[offset[t_id, 0] + n_tr_samples:offset[t_id + 1, 0]]

    x_s_tr, y_s_tr = shuffle(x_s_tr, y_s_tr)
    x_t_tr, y_t_tr = shuffle(x_t_tr, y_t_tr)
    x_s_tst, y_s_tst = shuffle(x_s_tst, y_s_tst)
    x_t_tst, y_t_tst = shuffle(x_t_tst, y_t_tst)

    y_s_tr[y_s_tr == -1] = 0
    y_t_tr[y_t_tr == -1] = 0
    y_s_tst[y_s_tst == -1] = 0
    y_t_tst[y_t_tst == -1] = 0  # 缺失标签用0补齐

    y_s_tr = to_one_hot(y_s_tr)
    y_t_tr = to_one_hot(y_t_tr)
    y_s_tst = to_one_hot(y_s_tst)
    y_t_tst = to_one_hot(y_t_tst)

    return x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst


def turn_tfidf(x):
    df = (x > 0.).sum(axis=0)
    idf = np.log(1. * len(x) / (df + 1))
    return np.log(1. + x) * idf[None, :]  # [None, :]可以起到reshape的作用？


if __name__ == '__main__':
    x, y, offset = load_amazon(5000, "data/amazon.mat")
    x_s_tr, y_s_tr, x_t_trs, y_t_trs, x_s_tst, y_s_tst, x_t_tsts, y_t_tsts = split_data(
        0, 1, x, y, offset, 2000)
    np.save('test2', x_t_tsts)

    # label = tf.constant([[1., 0], [0, 1.]], dtype=tf.float32)
    # logit = tf.constant([[0.8, 0.2], [0.1, 0.9]], dtype=tf.float32)
    # source_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)) \
    #               / (tf.reduce_sum(label) + tf.constant(1e-8, dtype=tf.float32))
    # sess = tf.Session()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    # print(sess.run(source_loss))