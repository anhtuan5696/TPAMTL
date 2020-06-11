import tensorflow as tf
import numpy as np
import pdb


class AMTL_LSTM(object):
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, config.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.output_keep_prob = 1
        self.state_keep_prob = 1
        self.input_keep_prob = 1

        self.global_step = tf.Variable(0, trainable=False)
        self.lr_decay = tf.train.exponential_decay(self.lr, self.global_step,
                                           10000, 0.8, staircase=False)

        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.train = tf.placeholder(dtype=tf.bool,name='train')
        # self.lr = config.LR

        self.build_model()


    def output(self, task_id, embed, beta_output):
        with tf.variable_scope("task_"+str(task_id)+'/output'):
            beta_att = tf.layers.dense(beta_output,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name='beta_att')

            c_i = tf.reduce_mean(beta_att * embed, 1)
            logits = tf.layers.dense(c_i,1,activation=None,use_bias=True,name='output_layer')

            self.beta_att_each.append(beta_att)

            return logits


    def build_model(self, use_lstm=True):
        print('Start building model')
        self.loss_each = []
        self.preds_each = []
        self.beta_att_each = []
        self.beta_main_each = []
        loss_task = 0
        self.att_each = {}
        self.transfer = {}
        self.test = []
        self.KL=[[0 for _ in range(self.num_tasks)] for i in range(self.num_tasks)]
        loss_B = 0
        self.beta_output = []

        with tf.variable_scope("B"):
            self.matrices = tf.get_variable("matrices",[self.num_tasks,self.num_tasks])
            reverse_identity = 1 - tf.eye(self.num_tasks)
            self.B_matrices = self.matrices * reverse_identity


        with tf.variable_scope("embed"):
            embed = tf.layers.dense(self.x,self.num_hidden,activation=None,use_bias=False)
        for task_id in range(self.num_tasks):
            with tf.variable_scope("task_"+str(task_id)+'/rnn'):
                def single_cell():            
                    lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
                    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, 
                                                         output_keep_prob=self.output_keep_prob,
                                                         input_keep_prob=self.input_keep_prob,
                                                         state_keep_prob=self.state_keep_prob,
                                                         dtype=tf.float32
                                                         )

                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
                beta_output, _ = tf.nn.dynamic_rnn(cell,
                                                       embed,
                                                       dtype=tf.float32)
            self.beta_output.append(beta_output)


        for task_id in range(self.num_tasks):
            # attention_sum 
            logits = self.output(task_id, embed, self.beta_output[task_id])
            preds = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[:,task_id:task_id+1])) 
            self.loss_each.append(loss)
            self.preds_each.append(preds)

            loss_task += loss 


        # amtl regularization
        all_params = []
        for task_id in range(self.num_tasks):
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task_%d/rnn'%(task_id))
            params = [tf.reshape(p,[1,-1]) for p in params]
            params = tf.concat(params,1)
            all_params.append(params)
        all_params = tf.concat(all_params,0)
        
        comb = tf.matmul(self.B_matrices,all_params)
        diff = all_params - comb
        loss_B = self.asym_lambda * tf.norm(diff,2)

        l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name or 'kernel' in v.name)]
        loss_l2 = self.l2_coeff*tf.reduce_sum(l2_losses)

        loss_B_regul = 0
        for task_id in range(self.num_tasks):
            loss_B_regul += self.asym_mu * self.loss_each[task_id] * tf.norm(self.B_matrices[:,task_id],1)

        self.loss_sum = loss_task + loss_B + loss_B_regul + loss_l2
        self.loss_all = {'loss_task':loss_task, 'loss_B': loss_B, 'loss_B_regul':loss_B_regul, 'loss_l2': loss_l2}

        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss_sum)
        print ('Model built')

