# import numpy as np
# import tensorflow as tf
# from scipy.io import loadmat
#
#
# def load_amazon(n_features, filepath):
#     """
#     Load Amazon Reviews
#     """
#     mat = loadmat(filepath)
#     xx = mat['xx']
#     yy = mat['yy']
#     offset = mat['offset']  # 这个数据文件的组织方式不熟悉
#
#     x = xx[:n_features, :].toarray().T  # n_samples * n_features
#     y = yy.ravel()  # 转一维数组
#
#     return x, y, offset
#
#
# def shuffle(x, y):
#     """Shuffle data"""
#     shuffled_id = np.arange(x.shape[0])
#     np.random.shuffle(shuffled_id)
#     x = x[shuffled_id, :]
#     y = y[shuffled_id]
#     return x, y
#
#
# def to_one_hot(a):
#     b = np.zeros((len(a), 2))
#     b[np.arange(len(a)), a] = 1
#     return b
#
#
# def split_data(s_id, t_id, x, y, offset, n_tr_samples, seed=0):
#     np.random.seed(seed)
#     x_s_tr = x[offset[s_id, 0]:offset[s_id, 0] + n_tr_samples, :]
#     x_t_tr = x[offset[t_id, 0]:offset[t_id, 0] + n_tr_samples, :]
#     x_s_tst = x[offset[s_id, 0] + n_tr_samples:offset[s_id + 1, 0], :]
#     x_t_tst = x[offset[t_id, 0] + n_tr_samples:offset[t_id + 1, 0], :]
#     y_s_tr = y[offset[s_id, 0]:offset[s_id, 0] + n_tr_samples]
#     y_t_tr = y[offset[t_id, 0]:offset[t_id, 0] + n_tr_samples]
#     y_s_tst = y[offset[s_id, 0] + n_tr_samples:offset[s_id + 1, 0]]
#     y_t_tst = y[offset[t_id, 0] + n_tr_samples:offset[t_id + 1, 0]]
#
#     x_s_tr, y_s_tr = shuffle(x_s_tr, y_s_tr)
#     x_t_tr, y_t_tr = shuffle(x_t_tr, y_t_tr)
#     x_s_tst, y_s_tst = shuffle(x_s_tst, y_s_tst)
#     x_t_tst, y_t_tst = shuffle(x_t_tst, y_t_tst)
#
#     y_s_tr[y_s_tr == -1] = 0
#     y_t_tr[y_t_tr == -1] = 0
#     y_s_tst[y_s_tst == -1] = 0
#     y_t_tst[y_t_tst == -1] = 0  # 缺失标签用0补齐
#
#     y_s_tr = to_one_hot(y_s_tr)
#     y_t_tr = to_one_hot(y_t_tr)
#     y_s_tst = to_one_hot(y_s_tst)
#     y_t_tst = to_one_hot(y_t_tst)
#
#     return x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst
#
#
# def turn_tfidf(x):
#     df = (x > 0.).sum(axis=0)
#     idf = np.log(1. * len(x) / (df + 1))
#     return np.log(1. + x) * idf[None, :]  # [None, :]可以起到reshape的作用？

import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def load_amazon(n_features, filepath):
    """
    Load Amazon Reviews
    """
    mat = loadmat(filepath)
    # print(mat.keys())
    xx = mat['xx']
    yy = mat['yy']
    offset = mat['offset']

    # 保留前n_features维特征，总共有27677个样本，offset存储每个领域的样本的起始位置
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


def generate(x, y, offset, id, n_tr_samples):
    x_tr = x[offset[id, 0]:offset[id, 0] + n_tr_samples, :]
    x_tst = x[offset[id, 0] + n_tr_samples:offset[id + 1, 0], :]
    y_tr = y[offset[id, 0]:offset[id, 0] + n_tr_samples]
    y_tst = y[offset[id, 0] + n_tr_samples:offset[id + 1, 0]]

    np.random.seed(0)  # 这里没有设置随机种子导致出错。。。。。这里要改，要把少量标注样本固定下来。。还是不对，需要固定每个目标域的50个标注样本
    x_tr, y_tr = shuffle(x_tr, y_tr)
    x_tst, y_tst = shuffle(x_tst, y_tst)

    y_tr[y_tr == -1] = 0
    y_tst[y_tst == -1] = 0
    y_tr = to_one_hot(y_tr)
    y_tst = to_one_hot(y_tst)

    return x_tr, y_tr, x_tst, y_tst


def split_data(s_id, t_ids, x, y, offset, n_tr_samples, seed=0):
    np.random.seed(seed)  # 设置随机种子的函数放这里没用

    x_s_tr, y_s_tr, x_s_tst, y_s_tst = generate(x, y, offset, s_id, n_tr_samples)

    x_t_trs, y_t_trs, x_t_tsts, y_t_tsts = [], [], [], []
    for id in t_ids:
        x_t_tr, y_t_tr, x_t_tst, y_t_tst = generate(x, y, offset, id, n_tr_samples)

        x_t_trs.append(x_t_tr)
        y_t_trs.append(y_t_tr)
        x_t_tsts.append(x_t_tst)
        y_t_tsts.append(y_t_tst)

    return x_s_tr, y_s_tr, x_t_trs, y_t_trs, x_s_tst, y_s_tst, x_t_tsts, y_t_tsts


def turn_tfidf(x):
    df = (x > 0.).sum(axis=0)
    idf = np.log(1. * len(x) / (df + 1))
    return np.log(1. + x) * idf[None, :]  # [None, :]可以起到reshape的作用？

if __name__ == '__main__':
    x, y, offset = load_amazon(5000, "data/amazon.mat")
    x_s_tr, y_s_tr, x_t_trs, y_t_trs, x_s_tst, y_s_tst, x_t_tsts, y_t_tsts = split_data(
        0, [1, 2, 3], x, y, offset, 2000)
    np.save('test1', np.array(x_t_trs))