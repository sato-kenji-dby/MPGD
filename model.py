import tensorflow as tf
import tensorflow.contrib.rnn as srnn
import numpy as np
from collections import deque
import random
from DNC import DNC
import math
import os


class DNC_PPO(object):
    replay_memory = deque()
    memory_size = 100
    def __init__(self, S_DIM, A_DIM, BATCH, A_UPDATE_STEPS, C_UPDATE_STEPS, num, METHOD, BETA, restore_dir=None):  # num: the index of user
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.BATCH = BATCH
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.decay = tf.placeholder(tf.float32, (), 'decay')
        self.a_lr = tf.placeholder(tf.float32, (), 'a_lr')
        self.c_lr = tf.placeholder(tf.float32, (), 'c_lr')
        self.global_step = tf.Variable(0, trainable=False)
        self.num = num
        self.METHOD = METHOD
        self.BETA = BETA
        # self.writer = tf.summary.FileWriter("./log/", self.sess.graph)

        # critic
        with tf.variable_scope('critic'):
            w1 = tf.Variable(tf.truncated_normal(
                [self.S_DIM, 210], stddev=0.01), name='w1')
            bias1 = tf.Variable(tf.constant(
                0.0, shape=[210], dtype=tf.float32), name='b1')
            l1 = tf.nn.relu(tf.matmul(self.tfs, w1) + bias1)

            w2 = tf.Variable(tf.truncated_normal(
                [210, 50], stddev=0.01), name='w2')
            bias2 = tf.Variable(tf.constant(
                0.0, shape=[50], dtype=tf.float32), name='b2')
            l2 = tf.nn.relu(tf.matmul(l1, w2) + bias2)
            if self.METHOD['DNC']:
                
                self.dnc = DNC(input_size=S_DIM, output_size=1,
                          seq_len=6, num_words=10, word_size=32, num_heads=1)
                #print("-----------------")
                #print(dnc.run(l2))
                self.v = tf.reshape(self.dnc.run(l2), [-1, np.shape(self.dnc.run(l2))[-1]])
            else:
                w3 = tf.Variable(tf.truncated_normal(
                    [50, 1], stddev=0.01), name='w3')
                bias3 = tf.Variable(tf.constant(
                    0.0, shape=[1], dtype=tf.float32), name='b3')
                self.v = tf.nn.relu(tf.matmul(l2, w3) + bias3)
            #print("--------------")
            #print(l2)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            # tf.summary.scalar(tf.reduce_mean(self.tfdc_r))
            self.advantage = self.tfdc_r - self.v


            self.closs = tf.reduce_mean(tf.square(self.advantage)) + \
                self.BETA  # * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w3))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.c_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.closs, vars_), 5.0)
            
            #print(self.closs)
            #print(tf.gradients(self.tfdc_r, vars_))
            #print(grads)
            #print(vars_)
            #print("self.v is ",self.v)

            self.ctrain_op = optimizer.apply_gradients(zip(grads, vars_))

        # actor
        pi, pi_params, l2_loss_a = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, _ = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            # choosing action  squeeze:reduce the first dimension
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(
                p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            elif METHOD['name'] == 'ddpg':
                self.aloss = -(tf.reduce_mean(pi.prob(self.tfa) * self.tfadv))
            elif METHOD['name'] == 'a2c':
                # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + tf.log(pi.scale)  # exploration
                self.aloss = -(tf.reduce_mean(pi.log_prob(self.tfa) * self.tfadv))
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv)) + \
                    self.BETA * l2_loss_a

        with tf.variable_scope('atrain'):
            # self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.a_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.aloss, vars_), 5.0)
            self.atrain_op = optimizer.apply_gradients(zip(grads, vars_), self.global_step)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        # self.summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('./tmp/vintf/', self.sess.graph)
        self.sess.run(init)
        if restore_dir:
            model_file = tf.train.latest_checkpoint(restore_dir)
            self.saver.restore(self.sess, model_file)

    def update(self, s, a, r, dec, alr, clr, epoch):
        self.sess.run(self.update_oldpi_op)
        if self.METHOD['DNC']:
            #print(s)
            adv = self.sess.run(
                self.advantage, {self.dnc.i_data: s, self.tfdc_r: r, self.decay: dec})
        else:
            adv = self.sess.run(
                self.advantage, {self.tfs: s, self.tfdc_r: r, self.decay: dec})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                aloss, _op, kl = self.sess.run(
                    [self.aloss, self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})
                if kl > 4 * self.METHOD['kl_target']:  # this in in google's paper
                    break
            # adaptive lambda, this is in OpenAI's paper
            if kl < self.METHOD['kl_target'] / 1.5:
                self.METHOD['lam'] /= 2
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
            # sometimes explode, this clipping is my solution
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)
        else:  # clipping method, find this is better (OpenAI's paper)
            for i in range(self.A_UPDATE_STEPS):
                aloss, _op = self.sess.run([self.aloss, self.atrain_op],
                                         {self.tfs: s, self.tfa: a, self.tfadv: adv, self.decay: dec, self.a_lr: alr, self.c_lr: clr})

        # update critic
        for i in range(self.C_UPDATE_STEPS):
            if self.METHOD['DNC']:
                closs, _op = self.sess.run([self.closs, self.ctrain_op], {
                    self.dnc.i_data: s, self.tfdc_r: r, self.decay: dec, self.a_lr: alr, self.c_lr: clr})
            else:
                closs, _op = self.sess.run([self.closs, self.ctrain_op], {
                                     self.tfs: s, self.tfdc_r: r, self.decay: dec, self.a_lr: alr, self.c_lr: clr})
        if epoch % 10 == 0:
            ckpt_path = "./ckpt/DNC_PPO_" + self.METHOD['name'] + '/' + str(self.num) + "/"
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            self.saver.save(self.sess, ckpt_path, global_step=epoch)
        return closs, aloss

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            w4 = tf.Variable(tf.truncated_normal(
                [self.S_DIM, 210], stddev=0.01), name='w4')
            bias4 = tf.Variable(tf.constant(
                0.0, shape=[210], dtype=tf.float32), name='b4')
            l3 = tf.nn.sigmoid(tf.matmul(self.tfs, w4) + bias4)
            # dnc = DNC(input_size=200, output_size=50,
            #           seq_len=0, num_words=10, word_size=4, num_heads=1)
            # l4 = tf.reshape(dnc.run(l3), [-1, np.shape(dnc.run(l3))[-1]])
            # print(np.shape(l4))

            w5 = tf.Variable(tf.truncated_normal(
                [210, 50], stddev=0.01), name='w5')
            bias5 = tf.Variable(tf.constant(
                0.0, shape=[50], dtype=tf.float32), name='b5')
            l4 = tf.nn.sigmoid(tf.matmul(l3, w5) + bias5)

            w6 = tf.Variable(tf.truncated_normal(
                [50, self.A_DIM], stddev=0.01), name='w6')
            bias6 = tf.Variable(tf.constant(
                0.0, shape=[self.A_DIM], dtype=tf.float32), name='b6')

            mu = 1 * tf.nn.sigmoid(tf.matmul(l4, w6) + bias6)
            # mu = 5 * tf.nn.sigmoid(tf.matmul(l4, w6) + bias6) + 0.0001
            # print('mu:', np.shape(mu))

            w7 = tf.Variable(tf.truncated_normal(
                [50, self.A_DIM], stddev=0.01), name='w7')
            bias7 = tf.Variable(tf.constant(
                0.0, shape=[self.A_DIM], dtype=tf.float32), name='b7')
            sigma = self.decay * \
                tf.nn.sigmoid(tf.matmul(l4, w7) + bias7) + 0.00001
            # print('sigma:',np.shape(sigma))

            # mu = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable)
            # sigma = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable) + 0.0001
            norm_dist = tf.distributions.Normal(
                loc=mu, scale=sigma)  # loc：mean  scale：sigma
        params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=name) 
        # tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
        l2_loss_a = 0
        return norm_dist, params, l2_loss_a

    def choose_action(self, s, dec):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, feed_dict={
            self.tfs: s, self.decay: dec})
        # a, sigma, mu = self.sess.run([self.sample_op, self.sigma, self.mu], feed_dict={self.tfs: s, self.decay: dec})

        return np.clip(a[0], 0.0001, 1)  # clip the output
    
    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        #print(s)
        #print(s.shape)
        #print(s[0])
        #print(self.v)
        #print({self.tfs: s})
        if self.METHOD['DNC']:
            return self.sess.run(self.v, {self.dnc.i_data: s})[0, 0]
        else:
            return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def loader(self):
        ckpt_path = "./ckpt/DNC_PPO_" + self.METHOD['name'] + '/' + str(self.num) + "/"
        model_file = tf.train.latest_checkpoint(ckpt_path)
        self.saver.restore(self.sess, model_file)