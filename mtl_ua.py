import tensorflow as tf
import numpy as np
import pdb

def attention_op(str_id, rnn_outputs, hidden_units, embed_size, steps):
    with tf.variable_scope(str_id+'_'):
        if str_id == 'alpha':
            p_att_shape = [hidden_units, 1]
        elif str_id == 'beta':
            p_att_shape = [hidden_units, embed_size]
        else:
            raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')

    #Create MU
    with tf.variable_scope(str_id+'MU'):
        mu_w = tf.Variable(tf.random_normal(p_att_shape, stddev=0.01), name='_mu')
        mu_b = tf.Variable(tf.zeros(p_att_shape[1], name='_mu'))
        mu =[]
        for _i in range(steps):
            mu_tmp = tf.matmul(rnn_outputs[:, _i, :], mu_w) + mu_b
            mu.append(mu_tmp)
        mu = tf.reshape(tf.concat(mu, 1), [-1, steps, p_att_shape[1]])

    #Create sigma
    with tf.variable_scope(str_id+'SIGMA'):
        sigma_w = tf.Variable(tf.random_normal(p_att_shape, stddev=0.01), name='sigma_weight')
        sigma_b = tf.Variable(tf.zeros(p_att_shape[1], name='sigma_bias'))
        sigma=[]
        for _k in range(steps):
            sigma_tmp = tf.matmul(rnn_outputs[:, _k, :], sigma_w) + sigma_b
            sigma.append(sigma_tmp)
        sigma = tf.reshape(tf.concat(sigma, 1), [-1, steps, p_att_shape[1]])
        sigma = tf.nn.softplus(sigma)

    distribution = tf.distributions.Normal(loc=mu, scale=sigma)
    att = distribution.sample([1])
    att = tf.squeeze(att, 0)

    if str_id == 'alpha':
        squashed_att = tf.nn.softmax(att, 1)
        print('Done with generating alpha attention.')
    elif str_id == 'beta':
        squashed_att = tf.nn.tanh(att)
        print('Done with generating beta attention.')
    else:
        raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')
    return squashed_att


class MTL_UA(object):
    dic = {}
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data') 

        self.y = tf.placeholder(shape=[None, self.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.train = tf.placeholder(dtype=tf.bool,name='train')
        self.keep_prob = 0.7
        # self.lr = config.LR

        self.global_step = tf.Variable(0, trainable=False)
        self.lr_decay = tf.train.exponential_decay(self.lr, self.global_step,
                                           10000, 0.8, staircase=False)

        self.build_model()

    def ua(self):
        with tf.variable_scope("ua", reuse=tf.AUTO_REUSE):
            def single_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
                return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, 
                                                     output_keep_prob=self.keep_prob,
                                                     dtype=tf.float32
                                                     )

            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

            with tf.variable_scope('embedded'):
                self.V = tf.get_variable('v_weight', shape=[self.num_features, self.num_hidden], dtype=tf.float32)

            v_emb = []
            with tf.variable_scope('embedded', reuse=True):
                for _j in range(self.num_steps):
                    self.V = tf.get_variable(name='v_weight')
                    embbed = tf.matmul(self.x[:, _j, :], self.V)
                    v_emb.append(embbed)
                self.embedded_v = tf.reshape(tf.concat(v_emb, 1), [-1, self.num_steps, self.num_hidden])

            #Reverse embedded_v
            reversed_v_outputs = tf.reverse(self.embedded_v, [1])

            with tf.variable_scope("myrnns_alpha") as scope:
                alpha_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                         reversed_v_outputs,
                                                         dtype=tf.float32
                                                         )

            with tf.variable_scope("myrnns_beta") as scope:
                beta_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                        reversed_v_outputs,
                                                        dtype=tf.float32
                                                        )

            #alpha
            alpha_embed_output = attention_op('alpha', alpha_rnn_outputs, self.num_hidden, self.num_hidden, self.num_steps)
            self.rev_alpha_embed_output = tf.reverse(alpha_embed_output, [1])

            #beta
            beta_embed_output = attention_op('beta', beta_rnn_outputs, self.num_hidden, self.num_hidden, self.num_steps)
            self.rev_beta_embed_output = tf.reverse(beta_embed_output, [1])

            # attention_multiplication
            c_i = tf.reduce_sum(self.rev_alpha_embed_output * (self.rev_beta_embed_output * self.embedded_v), 1)

        return c_i

    def output(self, task_id, output):
        with tf.variable_scope("decoder_"+str(task_id)):
            logits = tf.layers.dense(output,1,activation=None,use_bias=True,name='output_layer')

            return logits



    def build_model(self, use_lstm=True):
        print('Start building model')
        self.loss_each = []
        self.preds_each = []
        self.alpha_att_each = []
        self.beta_att_each = []
        loss_task = 0


        output = self.ua()
        for task_id in range(self.num_tasks):
            # device = task_id%self.num_gpus
            # with tf.device('/device:GPU:%d'%(device)):

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

