from multi_target_model import *
from dataset import *
import logging
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_load_from", "data/amazon.mat", "data path")
flags.DEFINE_integer("source_domain", 0, "source domain id")
flags.DEFINE_integer("target_domain", 1, "target domain id")
flags.DEFINE_list("target_domains", [1, 2, 3], "target domain ids")
flags.DEFINE_integer("valid_num", 500, "number of valid samples per domain")
flags.DEFINE_integer("labeled_num", 50, "number of labeled samples per domain")
flags.DEFINE_integer("tolerate_time", 20, "stop training if it exceeds tolerate time")
flags.DEFINE_integer("n_input", 5000, "size of input data")
flags.DEFINE_integer("n_classes", 2, "size of output data")
flags.DEFINE_integer("n_hidden_s", 50, "size of shared encoder hidden layer")
flags.DEFINE_integer("n_hidden_p", 50, "size of private encoder hidden layer")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_float("lr", 1e-4, "learning rate")

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='log/{0}_{1}.log'.format(FLAGS.source_domain, FLAGS.target_domain),
                    level=logging.WARNING, format=LOG_FORMAT)


class AdaptiveTrainer(object):
    def __init__(self):
        self.model_save_to = "output/model/{0}_to_{1}.pkl".format(FLAGS.source_domain, FLAGS.target_domain)

    def train_whole_model(self, old_batch, batch, x_valid, y_valid, x_test, y_test):
        """
        源域数据及目标域标注(或伪标签)数据训练整个模型。
        """
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())

            while True:
                R_loss, D_loss, C_loss, Diff_loss, P_loss, S_loss, train_accuracy = 0., 0., 0., 0., 0., 0., 0.
                for b in batch.generate(shuffle=True):
                    x, y, d = zip(*b)
                    _, r_loss = self.sess.run([model.R_solver, model.R_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, d_loss = self.sess.run([model.D_solver, model.D_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, p_loss = self.sess.run([model.P_solver, model.P_loss],
                                              feed_dict={model.X: x, model.Y: y, model.D: d})
                    _, s_loss, di_loss, accuracy, c_loss = self.sess.run(
                        [model.S_solver, model.S_loss, model.Diff_loss, model.acc, model.C_loss],
                        feed_dict={model.X: x, model.Y: y, model.D: d})
                    R_loss += r_loss
                    D_loss += d_loss
                    Diff_loss += di_loss
                    C_loss += c_loss
                    P_loss += p_loss
                    S_loss += s_loss
                    train_accuracy += accuracy
                print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, '
                      's_loss: {4}, diff_loss: {5}, train_acc: {6}'.format(
                    R_loss / batch.batch_num,
                    D_loss / batch.batch_num,
                    C_loss / batch.batch_num,
                    P_loss / batch.batch_num,
                    S_loss / batch.batch_num,
                    Diff_loss / batch.batch_num,
                    train_accuracy / batch.batch_num))

                if train_accuracy / batch.batch_num > 0.7:
                    # 在目标域验证集上做预测，用该目标域的50个标注样本生成原型，作为分类器
                    target_id = get_target_domain_id(FLAGS.target_domains, FLAGS.target_domain)
                    valid_accuracy = model.acc_test.eval({model.X: batch.x_t_list[target_id],
                                                          model.Y: batch.y_t_list[target_id],
                                                          model.test_X: x_valid, model.test_Y: y_valid},
                                                         session=self.sess)
                    # valid_accuracy = model.acc_test.eval({model.X: old_batch.x_s, model.Y: old_batch.y_s,
                    #                                       model.test_X: x_valid, model.test_Y: y_valid},
                    #                                      session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(sess=self.sess, save_path=self.model_save_to)
                    else:
                        wait_times += 1
                    if wait_times > FLAGS.tolerate_time:
                        print('best_result: {0}'.format(best_result))
                        break
                    print('valid_accuracy: {0}'.format(valid_accuracy))
            saver.restore(self.sess, self.model_save_to)
            target_id = get_target_domain_id(FLAGS.target_domains, FLAGS.target_domain)
            test_accuracy = model.acc_test.eval({model.X: batch.x_t_list[target_id],
                                                 model.Y: batch.y_t_list[target_id],
                                                 model.test_X: x_test, model.test_Y: y_test},
                                                session=self.sess)
            # test_accuracy = model.acc_test.eval({model.X: old_batch.x_s, model.Y: old_batch.y_s,
            #                                      model.test_X: x_test, model.test_Y: y_test},
            #                                     session=self.sess)
            logging.warning('valid_result: {0}, test_acc: {1}'.format(best_result, test_accuracy))
            print('test_accuracy: {0}'.format(test_accuracy))
            return best_result, test_accuracy

    def get_predictions(self, x, y, x_ts):
        """
        将每个目标域的样本转成概率分布
        """
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(FLAGS)
        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, save_path=self.model_save_to)
            probs = []
            for i in range(len(x_ts)):
                prob, = self.sess.run([model.prob],
                                      feed_dict={model.X: x[i], model.Y: y[i], model.test_X: x_ts[i]})
                # prob, = self.sess.run([model.prob],
                #                       feed_dict={model.X: x, model.Y: y, model.test_X: x_ts[i]})
                probs.append(prob)
        return probs

    # def select_samples(self, u_xs, probs):
    #     """
    #     由于不同领域的未标注数据需要打上不同的领域标签，所以需要分开处理
    #     :param u_xs:
    #     :param probs:
    #     :return:
    #     """
    #     x, y, d, unlabeled_xs = [], [], [], []
    #     for i in range(len(probs)):  # 对每个目标域的未标注样本和预测标签概率分布
    #         pos_idxes = set()
    #         neg_idxes = set()
    #         left_indexes = set(range(len(u_xs[i])))  # 序号集合
    #
    #         idxes = np.argsort(probs[i][:, 0])  # 按照0标签上元素的大小排列元素的序号
    #         end_idx = min(5, (probs[i][:, 0][idxes[:5]] < 0.5).sum())  # 每轮加入不超过10个样本
    #         begin_idx = min(5, (probs[i][:, 0][idxes[-5:]] > 0.5).sum())
    #         idx = min(end_idx, begin_idx)
    #         if idx == 0:
    #             idx = 1
    #         end_idx = idx
    #         begin_idx = idx
    #         neg_idxes.update(idxes[:end_idx])
    #         pos_idxes.update(idxes[-begin_idx:])
    #         left_indexes = left_indexes.intersection(idxes[end_idx: -begin_idx])
    #         pos_idxes = np.array(list(pos_idxes))
    #         neg_idxes = np.array(list(neg_idxes))
    #         left_indexes = np.array(list(left_indexes))
    #
    #         x_p = u_xs[i][pos_idxes]
    #         x_n = u_xs[i][neg_idxes]
    #         y_p = np.zeros(shape=(len(pos_idxes), 2), dtype='float32')
    #         y_p[:, 0] = 1.
    #         y_n = np.zeros(shape=(len(neg_idxes), 2), dtype='float32')
    #         y_n[:, 1] = 1.
    #         x.append(np.concatenate([x_p, x_n], axis=0))
    #         y.append(np.concatenate([y_p, y_n], axis=0))
    #         d.append(np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x_p) + len(x_n), 1)))
    #         unlabeled_x = u_xs[i][left_indexes]
    #         unlabeled_xs.append(unlabeled_x)
    #
    #         print('Pseudo label: {}'.format(len(x_p) + len(x_n)))
    #         print('Unlabeled samples: {}'.format(len(unlabeled_x)))
    #     return x, y, d, unlabeled_xs

    def select_samples(self, u_xs, probs):
        """
        不同领域的数据对应不同的领域标签，所以需要分开操作
        :param u_xs:
        :param probs:
        :return:
        """
        x, y, d, unlabeled_xs = [], [], [], []
        for i in range(len(probs)):  # 对每个目标域的未标注样本和预测标签概率分布
            pos_idxes = set()
            neg_idxes = set()
            left_indexes = set(range(len(u_xs[i])))

            idxes = np.argsort(probs[i][:, 0])
            end_idx = (probs[i][:, 0][idxes] < 0.2).sum()  # 预测负例个数
            begin_idx = (probs[i][:, 0][idxes] > 0.8).sum()  # 预测正例个数
            end_idx = end_idx if end_idx > 0 else 1  # 每次至少选两个
            begin_idx = begin_idx if begin_idx > 0 else 1

            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            left_indexes = left_indexes.intersection(idxes[end_idx: -begin_idx])
            pos_idxes = np.array(list(pos_idxes))
            neg_idxes = np.array(list(neg_idxes))
            left_indexes = np.array(list(left_indexes))

            x_p = u_xs[i][pos_idxes]
            x_n = u_xs[i][neg_idxes]
            y_p = np.zeros(shape=(len(pos_idxes), 2), dtype='float32')
            y_p[:, 0] = 1.
            y_n = np.zeros(shape=(len(neg_idxes), 2), dtype='float32')
            y_n[:, 1] = 1.
            x.append(np.concatenate([x_p, x_n], axis=0))
            y.append(np.concatenate([y_p, y_n], axis=0))
            d.append(np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x_p) + len(x_n), 1)))
            unlabeled_x = u_xs[i][left_indexes] if left_indexes.size else np.array([], dtype='float32')
            unlabeled_xs.append(unlabeled_x)

            print('Pseudo label: {}'.format(len(x_p) + len(x_n)))
            print('Unlabeled samples: {}'.format(len(unlabeled_x)))
        return x, y, d, unlabeled_xs

    def get_accuracy(self, x, y, x_test, y_test):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(FLAGS)
        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, save_path=self.model_save_to)
            accuracy = model.acc_test.eval({model.X: x, model.Y: y,
                                                 model.test_X: x_test, model.test_Y: y_test},
                                                session=self.sess)
            print('test_accuracy: {0}'.format(accuracy))
        return accuracy

    def get_embedding(self, x):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(FLAGS)
        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.total_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, save_path=self.model_save_to)
            emb_s, emb_p = self.sess.run([model.emb_s, model.emb_p], feed_dict={model.X: x})
        return emb_s, emb_p

    def train(self, batch, x_valid, y_valid, x_test, y_test):
        """
        self-training
        """
        best_result = 0.
        final_test_acc = 0.
        wait_times = 0

        new_batch = batch
        unlabeled_xs = np.copy(batch.x_ts)
        x_t_list, y_t_list, d_t_list = batch.x_t_list, batch.y_t_list, batch.d_t_list
        min_len = min([len(x) for x in batch.x_ts])

        while min_len > 0:
            print('Self-training...')
            valid_acc, test_acc = self.train_whole_model(batch, new_batch, x_valid, y_valid, x_test, y_test)
            probs = self.get_predictions(batch.x_t_list, batch.y_t_list, unlabeled_xs)
            # probs = self.get_predictions(batch.x_s, batch.y_s, unlabeled_xs)
            x_pseudo, y_pseudo, d_pseudo, unlabeled_xs = self.select_samples(unlabeled_xs, probs)
            x_t_list.extend(x_pseudo)
            y_t_list.extend(y_pseudo)
            d_t_list.extend(d_pseudo)
            new_batch = Batch(batch.x_s, batch.y_s, batch.d_s, x_t_list, y_t_list, d_t_list, unlabeled_xs, FLAGS.batch_size)
            min_len = min([len(x) for x in unlabeled_xs])

            if valid_acc > best_result:
                best_result = valid_acc
                final_test_acc = test_acc
                wait_times = 0
            else:
                wait_times += 1
            if wait_times > FLAGS.tolerate_time:
                print('best result: {}'.format(best_result))
                break
        print('Test accuracy: {}'.format(final_test_acc))

    # def train(self, batch, x_valid, y_valid, x_test, y_test):
    #     """
    #     不是严格的self-training
    #     """
    #     best_result = 0.
    #     final_test_acc = 0.
    #     wait_times = 0
    #     print('Pre-train model...')
    #     self.train_with_labeled_data(batch, x_valid, y_valid, x_test, y_test)
    #     probs = self.get_predictions(batch.x_t)
    #     x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
    #     while True:
    #         print('Self-training...')
    #         new_batch = Batch(batch.x_s, batch.y_s,  # 加入伪标签，生成新的train batch
    #                           np.concatenate([batch.x_t_tune, x_pseudo], axis=0),
    #                           np.concatenate([batch.y_t_tune, y_pseudo], axis=0),
    #                           batch.x_t, batch.batch_size)
    #         valid_acc, test_acc = self.train_with_labeled_data(new_batch, x_valid, y_valid, x_test, y_test)
    #         probs = self.get_predictions(batch.x_t)
    #         x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
    #         if valid_acc > best_result:
    #             best_result = valid_acc
    #             final_test_acc = test_acc
    #             wait_times = 0
    #         else:
    #             wait_times += 1
    #         if wait_times > self.FLAGS.tolerate_time:
    #             print('best result: {}'.format(best_result))
    #             break
    #     print('Test accuracy: {}'.format(final_test_acc))

    # def train(self, batch, x_valid, y_valid, x_test, y_test):
    #     wait_times = 0
    #     best_result = 0.
    #     self.graph = tf.Graph()
    #     tfConfig = tf.ConfigProto()
    #     tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    #     self.sess = tf.Session(graph=self.graph, config=tfConfig)
    #     model = AdaptiveModel(self.FLAGS)
    #
    #     with self.graph.as_default():
    #         model.build_model()
    #         saver = tf.train.Saver(var_list=model.total_theta)
    #         self.sess.run(tf.global_variables_initializer())
    #         # saver.restore(self.sess, self.model_load_from)
    #         while True:
    #             R_loss = 0.
    #             D_loss = 0.
    #             C_loss = 0.
    #             P_loss = 0.
    #             S_loss = 0.
    #             Diff_loss = 0.
    #             train_accuracy = 0.
    #             for b in batch.generate_pretrain_data(shuffle=True):
    #                 x, y, d = zip(*b)
    #                 _, r_loss = self.sess.run([model.R_solver, model.R_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, d_loss = self.sess.run([model.D_solver, model.D_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, c_loss = self.sess.run([model.C_solver, model.C_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, p_loss = self.sess.run([model.P_solver, model.P_loss],
    #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 _, s_loss, diff_loss, accuracy = self.sess.run(
    #                     [model.S_solver, model.S_loss, model.Diff_loss, model.accuracy],
    #                     feed_dict={model.X: x, model.Y: y, model.D: d})
    #                 R_loss += r_loss
    #                 D_loss += d_loss
    #                 C_loss += c_loss
    #                 P_loss += p_loss
    #                 S_loss += s_loss
    #                 Diff_loss += diff_loss
    #                 train_accuracy += accuracy
    #             # for b in batch.generate(domain='target', shuffle=True):
    #             #     x, y, d = zip(*b)
    #             #     _, r_loss = self.sess.run([model.R_solver_t, model.R_loss_t],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, d_loss = self.sess.run([model.D_solver, model.D_loss],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, c_loss = self.sess.run([model.C_t_solver, model.C_t_loss],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, p_loss = self.sess.run([model.P_t_solver, model.P_loss_t],
    #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
    #             #     _, s_loss, = self.sess.run([model.S_t_solver, model.S_t_loss],
    #             #                                feed_dict={model.X: x, model.Y: y, model.D: d})
    #             batch_nums = len(batch.x_s) / batch.batch_size
    #             print(batch_nums)
    #             print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, s_loss: {4}, acc: {5}'.format(
    #                 R_loss / batch_nums,
    #                 D_loss / batch_nums,
    #                 C_loss / batch_nums,
    #                 P_loss / batch_nums,
    #                 S_loss / batch_nums,
    #                 Diff_loss / batch_nums,
    #                 train_accuracy / batch_nums
    #             ))
    #             # print('train_loss: {0}, train_accuracy: {1}'.format(train_loss / batch_nums, train_accuracy / batch_nums))
    #             if train_accuracy / batch_nums > 0.:
    #                 valid_accuracy = model.accuracy.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 # pred = model.pred.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 # encoding = model.encoding.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
    #                 if valid_accuracy > best_result:
    #                     best_result = valid_accuracy
    #                     wait_times = 0
    #                     print('Save model...')
    #                     saver.save(sess=self.sess, save_path=self.model_save_to)
    #                 else:
    #                     wait_times += 1
    #                 if wait_times > self.FLAGS.tolerate_time:
    #                     print('best_result: {0}'.format(best_result))
    #                     break
    #                 # print('pred: {0}'.format(pred))
    #                 # print('encoding: {0}'.format(encoding))
    #                 print('valid_accuracy: {0}'.format(valid_accuracy))
    #         saver.restore(self.sess, self.model_save_to)
    #         test_accuracy = model.accuracy.eval({model.X: x_test, model.Y: y_test}, session=self.sess)
    #         print('test_accuracy: {0}'.format(test_accuracy))
    #         return test_accuracy


def get_target_domain_id(target_domains, target_domain):
    for i in range(len(target_domains)):
        if target_domains[i] == target_domain:
            return i
    return -1


def main(_):
    x, y, offset = load_amazon(5000, FLAGS.data_load_from)
    # 取出源域数据，和多个目标域数据
    x_s_tr, y_s_tr, x_t_trs, y_t_trs, x_s_tst, y_s_tst, x_t_tsts, y_t_tsts = split_data(
        FLAGS.source_domain, FLAGS.target_domains, x, y, offset, 2000)

    x_t_flat = []  # 多个目标域样本数组扁平化，统一计算tf-idf
    for i in range(len(x_t_trs)):
        x_t_flat.extend(x_t_trs[i])
        x_t_flat.extend(x_t_tsts[i])
    x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_flat]))
    x_s = x[:len(x_s_tr) + len(x_s_tst)]
    x_t = x[len(x_s):]
    x_s_tr = np.copy(x_s[:len(x_s_tr)])
    d_s_tr = np.tile(np.eye(len(FLAGS.target_domains) + 1)[0], (len(x_s_tr), 1))  # 源域数据的域标签

    # # 获取输入数据的编码表示
    # for i in range(len(x_t_trs)):
    #     x_t_tr = np.copy(x_t[:len(x_t_trs[i])])  # 目标域训练数据
    #     x_t = x_t[len(x_t_trs[i]):]
    #     x_t_tst = np.copy(x_t[:len(x_t_tsts[i])])  # 目标域测试数据
    #     x_t = x_t[len(x_t_tsts[i]):]
    #     y_t_tr = y_t_trs[i]
    #     d_t_tr = np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x_t_tr), 1))  # 第i个目标域的域标签
    #     # print(y_s_tr.shape)
    #     # print(y_t_tr.shape)
    #     x_s_tr = np.vstack([x_s_tr, x_t_tr])
    #     y_s_tr = np.vstack([y_s_tr, y_t_tr])
    #     d_s_tr = np.vstack([d_s_tr, d_t_tr])
    #
    # trainer = AdaptiveTrainer()
    # emb_xs, emb_xp = trainer.get_embedding(x_s_tr)
    # y = np.argmax(y_s_tr, axis=-1).reshape([-1, 1])
    # d = np.argmax(d_s_tr, axis=-1).reshape([-1, 1])
    # data = np.hstack([emb_xs, y, d])
    # np.save('shared_emb.npy', data)
    # data = np.hstack([emb_xp, y, d])
    # np.save('private_emb.npy', data)


    x_tune_list, y_tune_list, d_tune_list, x_trs = [], [], [], []
    # valid_x, valid_y = [], []
    for i in range(len(x_t_trs)):
        x_t_tr = np.copy(x_t[:len(x_t_trs[i])])  # 目标域训练数据
        x_t = x_t[len(x_t_trs[i]):]
        x_t_tst = np.copy(x_t[:len(x_t_tsts[i])])  # 目标域测试数据
        x_t = x_t[len(x_t_tsts[i]):]
        y_t_tst = y_t_tsts[i]
        d_t_tst = np.tile(np.eye(len(FLAGS.target_domains) + 1)[i + 1], (len(x_t_tst), 1))  # 第i个目标域的域标签
        x_tune_list.append(x_t_tst[:FLAGS.labeled_num])
        y_tune_list.append(y_t_tst[:FLAGS.labeled_num])
        d_tune_list.append(d_t_tst[:FLAGS.labeled_num])

        x_t_tst = x_t_tst[FLAGS.labeled_num:]
        y_t_tst = y_t_tst[FLAGS.labeled_num:]
        x_trs.append(x_t_tr)  # 只用各目标域训练集做自训练
        # valid_x.extend(x_t_tst[:FLAGS.valid_num])
        # valid_y.extend(y_t_tst[:FLAGS.valid_num])
        if FLAGS.target_domains[i] == FLAGS.target_domain:
            valid_x = x_t_tst[:FLAGS.valid_num]
            valid_y = y_t_tst[:FLAGS.valid_num]
            test_x = x_t_tst[FLAGS.valid_num:]
            test_y = y_t_tst[FLAGS.valid_num:]

    # valid_x = np.array(valid_x)
    # valid_y = np.array(valid_y)
    batch = Batch(x_s_tr, y_s_tr, d_s_tr, x_tune_list, y_tune_list, d_tune_list, x_trs, FLAGS.batch_size)
    trainer = AdaptiveTrainer()
    trainer.train(batch, valid_x, valid_y, test_x, test_y)


if __name__ == "__main__":
    tf.app.run()

# from multi_target_model import *
# from dataset import *
#
# flags = tf.flags
# FLAGS = flags.FLAGS
#
#
# class AdaptiveTrainer(object):
#     def __init__(self, flags):
#         self.FLAGS = flags
#         # self.model_load_from = "output/model/{0}_to_{1}_pre.pkl".format(flags.source_domain, flags.target_domain)
#         self.model_save_to = "output/model/{0}_to_{1}.pkl".format(flags.source_domain, flags.target_domain)
#
#     def train_whole_model(self, batch, x_valid, y_valid, x_test, y_test, classifier='source'):
#         """
#         用源域数据及目标域标注(或伪标签)数据预训练整个模型。
#         """
#         wait_times = 0
#         best_result = 0.
#         self.graph = tf.Graph()
#         tfConfig = tf.ConfigProto()
#         tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
#         self.sess = tf.Session(graph=self.graph, config=tfConfig)
#         model = AdaptiveModel(self.FLAGS)
#
#         with self.graph.as_default():
#             model.build_model()
#             saver = tf.train.Saver(var_list=model.total_theta)
#             self.sess.run(tf.global_variables_initializer())
#             if classifier == 'source':  # 预训练阶段需要用源域的分类器给目标域数据分类
#                 # model_path = self.model_load_from
#                 d_valid = np.tile(np.array([0., 1]), (len(x_valid), 1))
#                 d_test = np.tile(np.array([0., 1]), (len(x_test), 1))
#             else:
#                 # model_path = self.model_save_to
#                 d_valid = np.tile(np.array([1., 0]), (len(x_valid), 1))
#                 d_test = np.tile(np.array([1., 0]), (len(x_test), 1))
#                 # saver.restore(self.sess, save_path=self.model_load_from)  # 自适应过程每次都从预训练模型的参数点开始
#             while True:
#                 R_loss, D_loss, C_loss, Diff_loss, P_loss, S_loss = 0., 0., 0., 0., 0., 0.
#                 train_accuracy = 0.
#                 for b in batch.generate(shuffle=True):
#                     x, y, d = zip(*b)
#                     _, r_loss = self.sess.run([model.R_solver, model.R_loss],
#                                               feed_dict={model.X: x, model.Y: y, model.D: d})
#                     _, d_loss = self.sess.run([model.D_solver, model.D_loss],
#                                               feed_dict={model.X: x, model.Y: y, model.D: d})
#                     # _, c_loss = self.sess.run([model.C_solver, model.C_loss],
#                     #                           feed_dict={model.X: x, model.Y: y, model.D: d})
#                     _, p_loss = self.sess.run([model.P_solver, model.P_loss],
#                                               feed_dict={model.X: x, model.Y: y, model.D: d})
#                     _, s_loss, c_loss, di_loss, accuracy = self.sess.run(
#                         [model.S_solver, model.S_loss, model.C_loss, model.Diff_loss, model.acc],
#                         feed_dict={model.X: x, model.Y: y, model.D: d})
#                     R_loss += r_loss
#                     D_loss += d_loss
#                     C_loss += c_loss
#                     Diff_loss += di_loss
#                     P_loss += p_loss
#                     S_loss += s_loss
#                     train_accuracy += accuracy
#                 print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, s_loss: {4}, diff_loss: {5}, train_acc: {6}'.format(
#                     R_loss / batch.batch_num,
#                     D_loss / batch.batch_num,
#                     C_loss / batch.batch_num,
#                     P_loss / batch.batch_num,
#                     S_loss / batch.batch_num,
#                     Diff_loss / batch.batch_num,
#                     train_accuracy / batch.batch_num
#                 ))
#                 if train_accuracy / batch.batch_num > 0.7:
#                     valid_accuracy = model.acc.eval({model.X: x_valid, model.Y: y_valid, model.D: d_valid},
#                                                          session=self.sess)
#                     if valid_accuracy > best_result:
#                         best_result = valid_accuracy
#                         wait_times = 0
#                         print('Save model...')
#                         saver.save(sess=self.sess, save_path=self.model_save_to)
#                     else:
#                         wait_times += 1
#                     if wait_times > self.FLAGS.tolerate_time:
#                         print('best_result: {0}'.format(best_result))
#                         break
#                     print('valid_accuracy: {0}'.format(valid_accuracy))
#             saver.restore(self.sess, self.model_save_to)
#             test_accuracy = model.acc.eval({model.X: x_test, model.Y: y_test, model.D: d_test}, session=self.sess)
#             print('test_accuracy: {0}'.format(test_accuracy))
#             return best_result, test_accuracy
#
#     def get_predictions(self, x_t, classifier='source'):
#         self.graph = tf.Graph()
#         tfConfig = tf.ConfigProto()
#         tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
#         self.sess = tf.Session(graph=self.graph, config=tfConfig)
#         model = AdaptiveModel(self.FLAGS)
#         with self.graph.as_default():
#             model.build_model()
#             saver = tf.train.Saver(var_list=model.total_theta)
#             self.sess.run(tf.global_variables_initializer())
#             if classifier == 'source':
#                 # model_path = self.model_load_from
#                 d = np.tile(np.array([0., 1]), (len(x_t), 1))
#             else:
#                 # model_path = self.model_save_to
#                 d = np.tile(np.array([1., 0]), (len(x_t), 1))
#             saver.restore(self.sess, save_path=self.model_save_to)
#             probs, = self.sess.run([model.probs], feed_dict={model.X: x_t, model.D: d})
#         return probs
#
#     def select_samples(self, unlabeled_x, probs):
#         pos_idxes = set()
#         neg_idxes = set()
#         left_indexes = set(range(len(unlabeled_x)))
#         idxes = np.argsort(probs[:, 0])
#         end_idx = (probs[:, 0][idxes] < 0.3).sum()  # 预测负例个数
#         begin_idx = (probs[:, 0][idxes] > 0.7).sum()  # 预测正例个数
#         end_idx = end_idx if end_idx > 0 else 1  # 每次至少选两个
#         begin_idx = begin_idx if begin_idx > 0 else 1
#         neg_idxes.update(idxes[:end_idx])
#         pos_idxes.update(idxes[-begin_idx:])
#         left_indexes = left_indexes.intersection(idxes[end_idx: -begin_idx])
#
#         pos_idxes = np.array(list(pos_idxes))  # 正例的序号
#         neg_idxes = np.array(list(neg_idxes))  # 负例的序号
#         left_indexes = np.array(list(left_indexes))
#         x_p = unlabeled_x[pos_idxes]
#         x_n = unlabeled_x[neg_idxes]
#         y_p = np.zeros(shape=(len(pos_idxes), 2), dtype='float32')
#         y_p[:, 0] = 1.
#         y_n = np.zeros(shape=(len(neg_idxes), 2), dtype='float32')
#         y_n[:, 1] = 1.
#         x = np.concatenate([x_p, x_n], axis=0)
#         y = np.concatenate([y_p, y_n], axis=0)
#         unlabeled_x = unlabeled_x[left_indexes] if left_indexes.size else np.array([], dtype='float32')
#         print('Pseudo label: {}'.format(len(x)))
#         print('Unlabeled samples: {}'.format(len(unlabeled_x)))
#         return x, y, unlabeled_x
#
#     def train(self, batch, x_valid, y_valid, x_test, y_test):
#         """
#         self-training
#         """
#         best_result = 0.
#         final_test_acc = 0.
#         wait_times = 0
#
#         print('Pre-train model...')
#         self.train_whole_model(batch, x_valid, y_valid, x_test, y_test)
#         probs = self.get_predictions(batch.x_t)
#         x_pseudo, y_pseudo, unlabeled_x = self.select_samples(batch.x_t, probs)
#         x_t = np.concatenate([batch.x_t_tune, x_pseudo], axis=0)
#         y_t = np.concatenate([batch.y_t_tune, y_pseudo], axis=0)
#
#         while len(unlabeled_x) > 0:
#             print('Self-training...')
#             # 加入伪标签，生成新的train batch
#             new_batch = Batch(batch.x_s, batch.y_s, x_t, y_t, unlabeled_x, batch.batch_size)
#             valid_acc, test_acc = self.train_whole_model(new_batch, x_valid, y_valid, x_test, y_test, 'target')
#             probs = self.get_predictions(unlabeled_x, classifier='target')
#             x_pseudo, y_pseudo, unlabeled_x = self.select_samples(unlabeled_x, probs)
#             x_t = np.concatenate([x_t, x_pseudo], axis=0)
#             y_t = np.concatenate([y_t, y_pseudo], axis=0)
#
#             if valid_acc > best_result:
#                 best_result = valid_acc
#                 final_test_acc = test_acc
#                 wait_times = 0
#             else:
#                 wait_times += 1
#             if wait_times > self.FLAGS.tolerate_time:
#                 print('best result: {}'.format(best_result))
#                 break
#         print('Test accuracy: {}'.format(final_test_acc))
#
#     # def train(self, batch, x_valid, y_valid, x_test, y_test):
#     #     """
#     #     不是严格的self-training
#     #     """
#     #     best_result = 0.
#     #     final_test_acc = 0.
#     #     wait_times = 0
#     #     print('Pre-train model...')
#     #     self.train_with_labeled_data(batch, x_valid, y_valid, x_test, y_test)
#     #     probs = self.get_predictions(batch.x_t)
#     #     x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
#     #     while True:
#     #         print('Self-training...')
#     #         new_batch = Batch(batch.x_s, batch.y_s,  # 加入伪标签，生成新的train batch
#     #                           np.concatenate([batch.x_t_tune, x_pseudo], axis=0),
#     #                           np.concatenate([batch.y_t_tune, y_pseudo], axis=0),
#     #                           batch.x_t, batch.batch_size)
#     #         valid_acc, test_acc = self.train_with_labeled_data(new_batch, x_valid, y_valid, x_test, y_test)
#     #         probs = self.get_predictions(batch.x_t)
#     #         x_pseudo, y_pseudo = self.select_samples(batch.x_t, probs)
#     #         if valid_acc > best_result:
#     #             best_result = valid_acc
#     #             final_test_acc = test_acc
#     #             wait_times = 0
#     #         else:
#     #             wait_times += 1
#     #         if wait_times > self.FLAGS.tolerate_time:
#     #             print('best result: {}'.format(best_result))
#     #             break
#     #     print('Test accuracy: {}'.format(final_test_acc))
#
#     # def train(self, batch, x_valid, y_valid, x_test, y_test):
#     #     wait_times = 0
#     #     best_result = 0.
#     #     self.graph = tf.Graph()
#     #     tfConfig = tf.ConfigProto()
#     #     tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
#     #     self.sess = tf.Session(graph=self.graph, config=tfConfig)
#     #     model = AdaptiveModel(self.FLAGS)
#     #
#     #     with self.graph.as_default():
#     #         model.build_model()
#     #         saver = tf.train.Saver(var_list=model.total_theta)
#     #         self.sess.run(tf.global_variables_initializer())
#     #         # saver.restore(self.sess, self.model_load_from)
#     #         while True:
#     #             R_loss = 0.
#     #             D_loss = 0.
#     #             C_loss = 0.
#     #             P_loss = 0.
#     #             S_loss = 0.
#     #             Diff_loss = 0.
#     #             train_accuracy = 0.
#     #             for b in batch.generate_pretrain_data(shuffle=True):
#     #                 x, y, d = zip(*b)
#     #                 _, r_loss = self.sess.run([model.R_solver, model.R_loss],
#     #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
#     #                 _, d_loss = self.sess.run([model.D_solver, model.D_loss],
#     #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
#     #                 _, c_loss = self.sess.run([model.C_solver, model.C_loss],
#     #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
#     #                 _, p_loss = self.sess.run([model.P_solver, model.P_loss],
#     #                                           feed_dict={model.X: x, model.Y: y, model.D: d})
#     #                 _, s_loss, diff_loss, accuracy = self.sess.run(
#     #                     [model.S_solver, model.S_loss, model.Diff_loss, model.accuracy],
#     #                     feed_dict={model.X: x, model.Y: y, model.D: d})
#     #                 R_loss += r_loss
#     #                 D_loss += d_loss
#     #                 C_loss += c_loss
#     #                 P_loss += p_loss
#     #                 S_loss += s_loss
#     #                 Diff_loss += diff_loss
#     #                 train_accuracy += accuracy
#     #             # for b in batch.generate(domain='target', shuffle=True):
#     #             #     x, y, d = zip(*b)
#     #             #     _, r_loss = self.sess.run([model.R_solver_t, model.R_loss_t],
#     #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
#     #             #     _, d_loss = self.sess.run([model.D_solver, model.D_loss],
#     #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
#     #             #     _, c_loss = self.sess.run([model.C_t_solver, model.C_t_loss],
#     #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
#     #             #     _, p_loss = self.sess.run([model.P_t_solver, model.P_loss_t],
#     #             #                               feed_dict={model.X: x, model.Y: y, model.D: d})
#     #             #     _, s_loss, = self.sess.run([model.S_t_solver, model.S_t_loss],
#     #             #                                feed_dict={model.X: x, model.Y: y, model.D: d})
#     #             batch_nums = len(batch.x_s) / batch.batch_size
#     #             print(batch_nums)
#     #             print('r_loss: {0}, d_loss: {1}, c_loss: {2}, p_loss: {3}, s_loss: {4}, acc: {5}'.format(
#     #                 R_loss / batch_nums,
#     #                 D_loss / batch_nums,
#     #                 C_loss / batch_nums,
#     #                 P_loss / batch_nums,
#     #                 S_loss / batch_nums,
#     #                 Diff_loss / batch_nums,
#     #                 train_accuracy / batch_nums
#     #             ))
#     #             # print('train_loss: {0}, train_accuracy: {1}'.format(train_loss / batch_nums, train_accuracy / batch_nums))
#     #             if train_accuracy / batch_nums > 0.:
#     #                 valid_accuracy = model.accuracy.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
#     #                 # pred = model.pred.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
#     #                 # encoding = model.encoding.eval({model.X: x_valid, model.Y: y_valid}, session=self.sess)
#     #                 if valid_accuracy > best_result:
#     #                     best_result = valid_accuracy
#     #                     wait_times = 0
#     #                     print('Save model...')
#     #                     saver.save(sess=self.sess, save_path=self.model_save_to)
#     #                 else:
#     #                     wait_times += 1
#     #                 if wait_times > self.FLAGS.tolerate_time:
#     #                     print('best_result: {0}'.format(best_result))
#     #                     break
#     #                 # print('pred: {0}'.format(pred))
#     #                 # print('encoding: {0}'.format(encoding))
#     #                 print('valid_accuracy: {0}'.format(valid_accuracy))
#     #         saver.restore(self.sess, self.model_save_to)
#     #         test_accuracy = model.accuracy.eval({model.X: x_test, model.Y: y_test}, session=self.sess)
#     #         print('test_accuracy: {0}'.format(test_accuracy))
#     #         return test_accuracy
#
#
# def main(_):
#     x, y, offset = load_amazon(5000, FLAGS.data_load_from)
#     x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(
#         FLAGS.source_domain, FLAGS.target_domain, x, y, offset, 2000)  # 训练集2000样本
#
#     x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_tr, x_t_tst], axis=0))
#     x_s = x[:len(x_s_tr) + len(x_s_tst)]
#     x_t = x[len(x_s):]
#
#     x_s_tr = np.copy(x_s[:len(x_s_tr)])
#     x_s_tst = np.copy(x_s[len(x_s_tr):])  # 源域的测试集用不到
#
#     x_t_tr = np.copy(x_t[:len(x_t_tr)])  # train保持2000不变，test再切分为少样本、验证集和测试集
#     x_t_tst = np.copy(x_t[len(x_t_tr):])
#
#     x_t_tune = x_t_tst[:50]
#     y_t_tune = y_t_tst[:50]
#     x_t_tst = x_t_tst[50:]
#     y_t_tst = y_t_tst[50:]
#
#     x_t_valid = x_t_tst[:500]
#     y_t_valid = y_t_tst[:500]
#     x_t_tst = x_t_tst[500:]
#     y_t_tst = y_t_tst[500:]
#
#     batch = Batch(x_s_tr, y_s_tr, x_t_tune, y_t_tune, x_t_tr, FLAGS.batch_size)
#     trainer = AdaptiveTrainer(FLAGS)
#     trainer.train(batch, x_t_valid, y_t_valid, x_t_tst, y_t_tst)
#
#
# flags.DEFINE_string("data_load_from", "data/amazon.mat", "data path")
# flags.DEFINE_integer("source_domain", 0, "source domain id")
# flags.DEFINE_integer("target_domain", 2, "target domain id")
# flags.DEFINE_integer("n_domains", 2, "number of domains")
# flags.DEFINE_integer("tolerate_time", 20, "stop training if it exceeds tolerate time")
# flags.DEFINE_integer("n_input", 5000, "size of input data")
# flags.DEFINE_integer("n_classes", 2, "size of output data")
# flags.DEFINE_integer("n_hidden_s", 50, "size of shared encoder hidden layer")
# flags.DEFINE_integer("n_hidden_p", 50, "size of private encoder hidden layer")
# flags.DEFINE_integer("batch_size", 50, "batch size")
# flags.DEFINE_float("lr", 1e-4, "learning rate")
#
# if __name__ == "__main__":
#     tf.app.run()
