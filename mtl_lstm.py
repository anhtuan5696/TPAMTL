import tensorflow as tf
import numpy as np
import pdb


class MTL_LSTM(object):
    dic = {}
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data') 

        self.y = tf.placeholder(shape=[None, self.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.train = tf.placeholder(dtype=tf.bool,name='train')
        self.output_keep_prob = 1
        self.state_keep_prob = 1
        self.input_keep_prob = 1
        # self.lr = config.LR

        self.global_step = tf.Variable(0, trainable=False)
        self.lr_decay = tf.train.exponential_decay(self.lr, self.global_step,
                                           10000, 0.8, staircase=False)

        self.build_model()

    def shared_network(self):
        with tf.variable_scope("rnns", reuse=tf.AUTO_REUSE):
            def single_cell():            
                lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
                return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, 
                                                     output_keep_prob=self.output_keep_prob,
                                                     input_keep_prob=self.input_keep_prob,
                                                     state_keep_prob=self.state_keep_prob,
                                                     dtype=tf.float32
                                                     )

            

            with tf.variable_scope('embedded'):
                embed = tf.layers.dense(self.x,self.num_hidden,activation=None,use_bias=False)

                

            with tf.variable_scope("rnn_beta") as scope:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
                beta_output, _ = tf.nn.dynamic_rnn(cell,
                                                       embed,
                                                       dtype=tf.float32)
            

            beta_att = tf.nn.tanh(tf.layers.dense(beta_output,self.num_hidden,activation=None,use_bias=True,name='beta_att'))
            output = tf.reduce_mean(beta_att * embed, 1)
            self.beta_att_each.append(beta_att)

            return output

    def output(self, task_id, output):
        with tf.variable_scope("output_layer"+str(task_id)):
            logits = tf.layers.dense(output,1,activation=None,use_bias=True,name='output_layer')

            return logits



    def build_model(self, use_lstm=True):
        print('Start building model')
        self.loss_each = []
        self.preds_each = []
        self.alpha_att_each = []
        self.beta_att_each = []
        loss_task = 0


        output = self.shared_network()
        for task_id in range(self.num_tasks):

            logits = self.output(task_id, output)

            preds = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[:,task_id:task_id+1])) 
            self.loss_each.append(loss)
            self.preds_each.append(preds)

            loss_task += loss 

        l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name or 'kernel' in v.name)]
        loss_l2 = self.l2_coeff*tf.reduce_sum(l2_losses)

        self.loss_sum = loss_task + loss_l2
        self.loss_all = {'loss_task':loss_task, 'loss_l2':loss_l2}

        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss_sum)
        print ('Model built')

