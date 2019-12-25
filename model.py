import tensorflow as tf


class AdaptiveModel(object):
    def __init__(self, flags):
        self.FLAGS = flags

    def build_model(self):
        def matchnorm(x1, x2):  # L2范数？
            return tf.sqrt(tf.reduce_sum(tf.pow(x1 - x2, 2)))

        def scm(sx1, sx2, k):
            ss1 = tf.reduce_mean(tf.pow(sx1, k), axis=0)
            ss2 = tf.reduce_mean(tf.pow(sx2, k), axis=0)
            return matchnorm(ss1, ss2)

        def mmatch(x1, x2, n_moments):
            mx1 = tf.reduce_mean(x1, axis=0)
            mx2 = tf.reduce_mean(x2, axis=0)
            sx1 = x1 - mx1
            sx2 = x2 - mx2
            dm = matchnorm(mx1, mx2)
            scms = dm
            for i in range(n_moments - 1):
                scms += scm(sx1, sx2, i + 2)
            return scms

        # tf graph input
        self.X_s_u = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.X_t_u = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.X_s = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.Y_s = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])
        self.X_t = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_input])
        self.Y_t = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS.n_classes])

        # fd network
        self.common_encode_mlp = MLP('common_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_c],
                                     [tf.nn.sigmoid])
        self.target_encode_mlp = MLP('target_encode_mlp', [self.FLAGS.n_input, self.FLAGS.n_hidden_t],
                                     [tf.nn.sigmoid])
        self.common_decode_mlp = MLP('common_decode_mlp',
                                     [self.FLAGS.n_hidden_c, (self.FLAGS.n_hidden_c + self.FLAGS.n_input) // 2,
                                      self.FLAGS.n_input],
                                     [tf.nn.tanh, tf.nn.relu])
        self.target_decode_mlp = MLP('target_decode_mlp',
                                     [self.FLAGS.n_hidden_t, (self.FLAGS.n_hidden_t + self.FLAGS.n_input) // 2,
                                      self.FLAGS.n_input],
                                     [tf.nn.tanh, tf.nn.relu])
        self.common_output_mlp = MLP('common_output_mlp', [self.FLAGS.n_hidden_c, self.FLAGS.n_classes],
                                     [identity])
        self.target_output_mlp = MLP('target_output_mlp', [self.FLAGS.n_hidden_t, self.FLAGS.n_classes],
                                     [identity])

        # Get CMD loss
        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 3)

        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))  # 平方和损失

        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.pred_s = pred_s
        self.prob_s = tf.nn.softmax(pred_s)
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_s, logits=pred_s))
        correct_prediction = tf.equal(tf.argmax(self.Y_s, axis=1), tf.argmax(pred_s, axis=1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        # Get target classification loss
        encoding_t_t = self.target_encode_mlp.apply(self.X_t)
        pred_t = self.target_output_mlp.apply(encoding_t_t)
        self.pred_t = pred_t
        self.prob_t = tf.nn.softmax(pred_t)
        self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_t, logits=pred_t))
        correct_prediction = tf.equal(tf.argmax(self.Y_t, axis=1), tf.argmax(pred_t, axis=1))
        self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.common_output_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.target_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:  # 找不到W，返回-1，于是对所有参数计算正则化项
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        for tensor in self.target_output_mlp.parameters + self.target_encode_mlp.parameters:
            if tensor.name.find('W') != 0:
                l2_norm += 4 * tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.FLAGS.recon * self.R_loss +
                     self.FLAGS.alpha * self.C_loss +
                     self.FLAGS.belta * self.T_loss +
                     self.FLAGS.gamma * self.cmd_c_loss +
                     self.FLAGS.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.loss, var_list=self.theta)

        self.common_loss = (self.FLAGS.recon * self.R_loss +
                            self.FLAGS.alpha * self.C_loss +
                            self.FLAGS.gamma * self.cmd_c_loss +
                            self.FLAGS.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        # self.common_loss = self.C_loss
        self.common_solver = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.common_loss, var_list=self.theta)

        self.target_loss = (self.FLAGS.recon * self.R_loss +
                            self.FLAGS.belta * self.T_loss +
                            self.FLAGS.gamma * self.cmd_c_loss +
                            self.FLAGS.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        self.target_solver = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr).minimize(loss=self.target_loss, var_list=self.theta)


class MLP(object):
    def __init__(self, name, dims, activations):
        self.name = name
        self.dims = dims
        self.activations = activations
        self.weights = []
        self.biases = []
        self._initialize()

    @property
    def parameters(self):
        return self.weights + self.biases

    def _initialize(self):
        for i in range(len(self.dims) - 1):
            w = tf.Variable(xavier_init([self.dims[i], self.dims[i + 1]]), name=self.name + '_w_{0}'.format(i))
            b = tf.Variable(xavier_init([self.dims[i + 1]]), name=self.name + '_b_{0}'.format(i))
            self.weights.append(w)
            self.biases.append(b)

    def apply(self, x):
        out = x
        for a, w, b in zip(self.activations, self.weights, self.biases):
            out = a(tf.add(tf.matmul(out, w), b))
        return out


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # xavier初始化方法的标准差计算公式，置out_dim=0
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def identity(x):
    return x
