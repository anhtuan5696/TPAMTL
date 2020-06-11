import tensorflow as tf
import numpy as np
import pdb


class MTL_TRANS(object):
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, config.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.train = tf.placeholder(dtype=tf.bool,name='train')
        self.output_keep_prob = 1
        self.state_keep_prob = 1
        self.input_keep_prob = 1


        self.global_step = tf.Variable(0, trainable=False)
        self.lr_decay = tf.train.exponential_decay(self.lr, self.global_step,
                                           10000, 0.8, staircase=False)
        self.build_model()


    def output(self, task_id, embed, beta_output):
        with tf.variable_scope("task_"+str(task_id)+'/output'):
            #beta_att = tf.layers.dense(beta_output,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name='beta_att')
            #c_i = tf.reduce_mean(beta_att * embed, 1)

            c_i = tf.reduce_mean(beta_output, 1)
            logits = tf.layers.dense(c_i,1,activation=None,use_bias=True,name='output_layer')


            return logits


    def attention(self,q,k,v):
        k_transposed = tf.transpose(k,perm=[0,2,1])
        att = tf.matmul(q,k_transposed) / tf.sqrt(tf.to_float(tf.shape(k)[2]))
        att = tf.nn.softmax(att,2)
        attention = tf.matmul(att,v)
        return attention

    def multihead(self,multihead_input):

        # multihead sub-layer
        # multihead
        d_att = int(self.num_hidden / 4)
        multihead_output = []
        for i in range(4):
            with tf.variable_scope("head"+str(i)):
                q_i = tf.layers.dense(inputs=multihead_input,
                                      units=d_att,
                                      activation=None,
                                      use_bias=False)
                k_i = tf.layers.dense(inputs=multihead_input,
                                      units=d_att,
                                      activation=None,
                                      use_bias=False)
                v_i = tf.layers.dense(inputs=multihead_input,
                                      units=d_att,
                                      activation=None,
                                      use_bias=False)
                multihead_output.append(self.attention(q_i,k_i,v_i))

        multihead_output = tf.concat(multihead_output,2)

        # residual and norm of multihead layer
        multihead_output = tf.contrib.layers.layer_norm(multihead_output+multihead_input)



        # feed-forward sub-layer:
        # 2 feed forward layers
        ff1 = tf.contrib.layers.fully_connected(
                                                inputs=multihead_output,
                                                num_outputs=self.num_hidden,
                                                activation_fn=tf.nn.relu)
        ff2 = tf.contrib.layers.fully_connected(
                                                inputs=ff1,
                                                num_outputs=self.num_hidden,
                                                activation_fn=tf.identity)

        # residual
        ffn_output = tf.contrib.layers.layer_norm(ff2+multihead_output)


        return ffn_output

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
        self.att_loc = {}
        self.att_scale = {}
        self.att_each = {}
        self.att_each_before = {}
        self.test = {}

        self.beta_output = []


        with tf.variable_scope("embed"):
            embed = tf.layers.dense(self.x,self.num_hidden,activation=None,use_bias=False)
            # positional encoding: sin and cos
            positional_encoding = [
                                    [np.sin(pos/10000**(i/self.num_hidden)) if i%2==0 else np.cos(pos/10000**((i-1)/self.num_hidden)) for i in range(self.num_hidden)]
                                  for pos in range(self.num_steps)]
            positional_encoding = tf.convert_to_tensor(positional_encoding)

            embed = embed + positional_encoding


        f = embed
        with tf.variable_scope('shared/trans'):
            for i in range(1):            
                with tf.variable_scope("layer"+str(i)):
                    f = self.multihead(f)


        for task_id in range(self.num_tasks):

            # attention_sum 
            logits = self.output(task_id, embed, f)
            preds = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[:,task_id:task_id+1])) 
            self.loss_each.append(loss)
            self.preds_each.append(preds)

            loss_task += loss 


        l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name or 'kernel' in v.name)]
        loss_l2 = self.l2_coeff*tf.reduce_sum(l2_losses) 

        self.loss_sum = loss_task + loss_l2
        self.loss_all = {'loss_task':loss_task, 'loss_l2': loss_l2}
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss_sum)
        print ('Model built')

