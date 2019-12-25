import tensorflow as tf
import numpy as np
from pathlib import Path
from dataset import *
from model import *
import logging

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_load_from", "data/amazon.mat", "data path")
flags.DEFINE_integer("source_domain", 0, "source domain id")
flags.DEFINE_integer("target_domain", 2, "target domain id")
flags.DEFINE_list("target_domains", [2, 1, 3], "target domain ids")
flags.DEFINE_integer("valid_num", 500, "number of valid samples per domain")
flags.DEFINE_integer("labeled_num", 50, "number of labeled samples per domain")
flags.DEFINE_integer("tolerate_time", 20, "stop training if it exceeds tolerate time")
flags.DEFINE_integer("n_input", 5000, "size of input data")
flags.DEFINE_integer("n_classes", 2, "size of output data")
flags.DEFINE_integer("n_hidden_c", 50, "size of common encoder hidden layer")
flags.DEFINE_integer("n_hidden_t", 50, "size of target encoder hidden layer")
flags.DEFINE_float("lr", 5e-3, "learning rate")
flags.DEFINE_float("recon", 1., "the coefficient of reconstruction loss")
flags.DEFINE_float("alpha", 1., "the coefficient of common classification loss")
flags.DEFINE_float("belta", 1., "the coefficient of target classification loss")
flags.DEFINE_float("gamma", 1., "the coefficient of common CMD loss")
flags.DEFINE_float("lamb", 1., "the coefficient of target CMD loss")

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='{0}_{1}_beta.log'.format(FLAGS.source_domain, FLAGS.target_domain),
                    level=logging.WARNING, format=LOG_FORMAT)


class CoTrainer(object):
    def __init__(self, flags, **kwargs):
        self.FLAGS = flags
        self.common_model_save_to = "output/model/common_{0}_to_{1}.pkl".format(flags.source_domain,
                                                                                flags.target_domain)
        self.target_model_save_to = "output/model/target_{0}_to_{1}.pkl".format(flags.source_domain,
                                                                                flags.target_domain)

    def train_common_model(self, x_s_u, x_t_u, x_s, y_s, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, c_loss, cmd_c_loss, cmd_t_loss, r_loss, accuracy) = self.sess.run(
                    [model.common_solver, model.C_loss, model.cmd_c_loss, model.cmd_t_loss, model.R_loss,
                     model.accuracy_s],
                    feed_dict={model.X_s: x_s,  # 没有按batch输入也没有规定迭代多少轮，收敛为止
                               model.Y_s: y_s,
                               model.X_s_u: x_s_u,
                               model.X_t_u: x_t_u})

                print('accuracy_s: {0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = model.accuracy_s.eval({model.X_s: x_valid,
                                                            model.Y_s: y_valid},
                                                           session=self.sess)
                    pred_s = model.pred_s.eval({model.X_s: x_valid,
                                                model.Y_s: y_valid},
                                               session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, self.common_model_save_to)
                    else:
                        wait_times += 1
                    if wait_times >= self.FLAGS.tolerate_time:
                        print('best_result: {0}'.format(best_result))
                        break
                    # print('pred_s: {0}'.format(pred_s))
                    print('valid_accuracy: {0}'.format(valid_accuracy))
            saver.restore(self.sess, self.common_model_save_to)
            test_accuracy = model.accuracy_s.eval({model.X_s: x_test,
                                                   model.Y_s: y_test},
                                                  session=self.sess)  # 仅用目标域数据做测试
            print('test_accuracy: {0}'.format(test_accuracy))
            return best_result

    def train_target_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, t_loss, cmd_c_loss, cmd_t_loss, r_loss, accuracy) = self.sess.run(
                    [model.target_solver, model.T_loss, model.cmd_c_loss, model.cmd_t_loss, model.R_loss,
                     model.accuracy_t],
                    feed_dict={model.X_t: x_t,
                               model.Y_t: y_t,
                               model.X_s_u: x_s_u,
                               model.X_t_u: x_t_u})
                print('accuracy_t: {0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = model.accuracy_t.eval({model.X_t: x_valid,
                                                            model.Y_t: y_valid},
                                                           session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, self.target_model_save_to)
                    else:
                        wait_times += 1
                    if wait_times >= self.FLAGS.tolerate_time:
                        print('best_result: {0}'.format(best_result))
                        break
                    print('valid_accuracy: {0}'.format(valid_accuracy))
            saver.restore(self.sess, self.target_model_save_to)
            test_accuracy = model.accuracy_t.eval({model.X_t: x_test,
                                                   model.Y_t: y_test},
                                                  session=self.sess)
            print('test_accuracy: {0}'.format(test_accuracy))
            return best_result

    def train_combined_model(self, x_valid, y_valid, x_test, y_test):  # 只是跑个加权的测试结果
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)

        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            common_valid_probs, = self.sess.run([model.prob_s], feed_dict={model.X_s: x_valid, model.Y_s: y_valid})
            common_test_probs, = self.sess.run([model.prob_s], feed_dict={model.X_s: x_test, model.Y_s: y_test})
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            target_valid_probs, = self.sess.run([model.prob_t], feed_dict={model.X_t: x_valid, model.Y_t: y_valid})
            target_test_probs, = self.sess.run([model.prob_t],
                                               feed_dict={model.X_t: x_test, model.Y_t: y_test})  # 逗号不能扔啊

            best_result = 0.
            best_beta = 0.
            for beta in np.arange(0, 1.1, 0.1):
                valid_probs = common_valid_probs + beta * target_valid_probs
                valid_accuracy = np.equal(valid_probs.argmax(axis=1), y_valid.argmax(axis=1)).mean()
                if valid_accuracy > best_result:
                    best_result = valid_accuracy
                    best_beta = beta
            valid_accuracy = best_result
            test_probs = common_test_probs + best_beta * target_test_probs
            test_accuracy = np.equal(test_probs.argmax(axis=1), y_test.argmax(axis=1)).mean()
            print('valid_accuracy: {0}'.format(valid_accuracy))
            print('test_accuracy: {0}'.format(test_accuracy))
            logging.warning('best_beta: {0}, test_acc: {1}'.format(best_beta, test_accuracy))
            return valid_accuracy, test_accuracy

    def train(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        # 无标签特征表示用于计算CMD度量和重建任务，有监督部分（除了目标域少量样本外，都是前面的子集）用于Co-training
        U = np.copy(x_t_u)
        select_num = 5  # 每一轮加入到训练集的样本个数（每个分类器）不超过5个
        best_result = 0.
        final_test_accuracy = 0.
        wait_times = 0

        while len(U) > 0:
            print('Train common model...')
            self.train_common_model(x_s_u, x_t_u, np.concatenate([x_s, x_t]), np.concatenate([y_s, y_t]),
                                    x_valid, y_valid, x_test, y_test)
            print('Train target model...')
            self.train_target_model(x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test)

            # Select U
            probs = [self.get_common_prediction(U), self.get_target_prediction(U)]
            x_hat, y_hat, U = self.select_samples(U, probs, select_num)
            x_t = np.concatenate([x_t, x_hat], axis=0)  # 这里更新x_t之后，在上面训练common和target模型会更新标注数据
            y_t = np.concatenate([y_t, y_hat], axis=0)
            print('Test combined model...')
            valid_accuracy, test_accuracy = self.train_combined_model(x_valid, y_valid, x_test, y_test)

            if valid_accuracy > best_result:
                best_result = valid_accuracy
                final_test_accuracy = test_accuracy  # 验证集上取得最好结果的参数，在测试集上的结果
                wait_times = 0
            else:
                wait_times += 1
            if wait_times >= self.FLAGS.tolerate_time:
                print('best result:{0}'.format(best_result))
                break
        print('Test accuracy:{0}'.format(final_test_accuracy))

    # def train(self, x_s_u, x_t_u, x_s, y_s, x_valid, y_valid, x_test, y_test):
    #     print('Train common model...')
    #     self.train_common_model(x_s_u, x_t_u, x_s, y_s, x_valid, y_valid, x_test, y_test)

    def select_samples(self, U, probs, select_num):
        neg_idxes = set()
        pos_idxes = set()
        left_idxes = set(range(len(U)))  # 未选中的样本
        for prob in probs:  # 每轮选择的正负例不超过5个
            idxes = np.argsort(prob[:, 0])  # 在类别0上的预测结果按照从小到大排序的元素的原始位置
            end_idx = min(select_num, (prob[:, 0][idxes[:select_num]] < 0.5).sum())
            begin_idx = min(select_num, (prob[:, 0][idxes[-select_num:]] > 0.5).sum())
            idx = min(end_idx, begin_idx)
            if idx == 0:
                idx = 1
            end_idx = idx
            begin_idx = idx
            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            print('pos num: ', len(pos_idxes))
            print('neg num: ', len(neg_idxes))
            left_idxes = left_idxes.intersection(idxes[end_idx: -begin_idx])

        pos_idxes = np.array(list(pos_idxes))
        neg_idxes = np.array(list(neg_idxes))
        left_idxes = np.array(list(left_idxes))
        x_p = U[pos_idxes]
        x_n = U[neg_idxes]
        y_p = np.zeros(shape=(len(pos_idxes), 2), dtype='float32')
        y_p[:, 0] = 1.
        y_n = np.zeros(shape=(len(neg_idxes), 2), dtype='float32')
        y_n[:, 1] = 1.
        x = np.concatenate([x_p, x_n], axis=0)
        y = np.concatenate([y_p, y_n], axis=0)
        x, y = shuffle(x, y)
        U = U[left_idxes]
        print('unlabeled num: ', len(U))
        print(len(left_idxes))
        return x, y, U

    def get_common_prediction(self, X):
        self.graph = tf.Graph()  # 新建空白图，而不是变量重用
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)
        with self.graph.as_default():
            model.build_model()
            saver = tf.train.Saver(var_list=model.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            probs, = self.sess.run([model.prob_s], feed_dict={model.X_s: X})  # 这个逗号不能扔啊！
        return probs

    def get_target_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        model = AdaptiveModel(self.FLAGS)
        with self.graph.as_default():
            model.build_model()  # 这里不要忘了
            saver = tf.train.Saver(var_list=model.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            probs, = self.sess.run([model.prob_t], feed_dict={model.X_t: X})
        return probs


def main(_):
    # x, y, offset = load_amazon(5000, FLAGS.data_load_from)
    # x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(
    #     FLAGS.source_domain, FLAGS.target_domain, x, y, offset, 2000)  # 训练集2000样本
    #
    # x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_tr, x_t_tst], axis=0))
    # x_s = x[:len(x_s_tr) + len(x_s_tst)]
    # x_t = x[len(x_s):]
    #
    # x_s_tr = np.copy(x_s[:len(x_s_tr)])
    # x_s_tst = np.copy(x_s[len(x_s_tr):])  # 源域的测试集用不到
    #
    # x_t_tr = np.copy(x_t[:len(x_t_tr)])  # train保持2000不变，test再切分为少样本、验证集和测试集
    # x_t_tst = np.copy(x_t[len(x_t_tr):])
    #
    # x_t_tune = x_t_tst[:50]
    # y_t_tune = y_t_tst[:50]
    # x_t_tst = x_t_tst[50:]
    # y_t_tst = y_t_tst[50:]
    #
    # x_t_valid = x_t_tst[:500]
    # y_t_valid = y_t_tst[:500]
    # x_t_tst = x_t_tst[500:]
    # y_t_tst = y_t_tst[500:]

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

    x_t_tune, y_t_tune, x_t_tr = [], [], []
    for i in range(len(x_t_trs)):
        _x_t_tr = np.copy(x_t[:len(x_t_trs[i])])  # 目标域训练数据
        x_t = x_t[len(x_t_trs[i]):]
        x_t_tst = np.copy(x_t[:len(x_t_tsts[i])])  # 目标域测试数据
        x_t = x_t[len(x_t_tsts[i]):]
        y_t_tst = y_t_tsts[i]
        x_t_tune.extend(x_t_tst[:FLAGS.labeled_num])
        y_t_tune.extend(y_t_tst[:FLAGS.labeled_num])

        x_t_tst = x_t_tst[FLAGS.labeled_num:]
        y_t_tst = y_t_tst[FLAGS.labeled_num:]
        if FLAGS.target_domains[i] == FLAGS.target_domain:
            valid_x = x_t_tst[:FLAGS.valid_num]
            valid_y = y_t_tst[:FLAGS.valid_num]
            test_x = x_t_tst[FLAGS.valid_num:]
            test_y = y_t_tst[FLAGS.valid_num:]
            x_t_tr.extend(_x_t_tr)  # 只用各目标域训练集生成伪标签
        else:
            x_t_tr.extend(_x_t_tr)

    trainer = CoTrainer(FLAGS)
    trainer.train(x_s_tr, x_t_tr, np.copy(x_s_tr), y_s_tr, x_t_tune, y_t_tune, valid_x, valid_y, test_x, test_y)
    # trainer.train(x_s_tr, x_t_tr, np.copy(x_s_tr), y_s_tr, x_t_tune, y_t_tune, x_t_valid, y_t_valid, x_t_tst, y_t_tst)
    # trainer.train(x_s_tr, x_t_tr, np.copy(x_s_tr), y_s_tr, x_t_valid, y_t_valid, x_t_tst, y_t_tst)


if __name__ == "__main__":
    tf.app.run()

    # test1 = np.load('valid1.npy')
    # test2 = np.load('v1.npy')
    # for i in range(len(test1)):
    #     if int(sum(test1[i])) != int(sum(test2[i])):
    #         print('error')
# 3915 3036 5131 5395