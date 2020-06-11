import tensorflow as tf
import numpy as np
import pdb


class TP_AMTL_2(object):
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, config.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.train = tf.placeholder(dtype=tf.bool,name='train')
        self.keep_prob = 1


        self.global_step = tf.Variable(0, trainable=False)
        self.lr_decay = tf.train.exponential_decay(self.lr, self.global_step,
                                           10000, 0.8, staircase=False)
        self.build_model()


    def output(self, task_id, embed, beta_output):
        with tf.variable_scope("task_"+str(task_id)+'/output'):
            beta_att = tf.layers.dense(beta_output,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name='beta_att')

            c_i = tf.reduce_mean(beta_att * embed, 1)
            logits = tf.layers.dense(c_i,1,activation=None,use_bias=True,name='output_layer')

            self.beta_att_each.append(beta_att)

            return logits

    def transfer(self,Q,K,V,to_task,from_task):
        with tf.variable_scope('transfer_to%d_from%d'%(to_task,from_task),reuse=tf.AUTO_REUSE):
            helper = np.zeros([self.num_steps,self.num_steps])
            for i in range(self.num_steps):
                for j in range(self.num_steps):
                    if i>=j:
                        helper[i][j] = 1.0
            helper = tf.convert_to_tensor(helper,dtype=tf.float32)

            QW = tf.layers.dense(Q,self.num_hidden,activation=tf.nn.relu,use_bias=True)
            QW = tf.layers.dense(QW,self.num_hidden,activation=tf.nn.relu,use_bias=True)
            KW = tf.layers.dense(K,self.num_hidden,activation=tf.nn.relu,use_bias=True)
            KW = tf.layers.dense(KW,self.num_hidden,activation=tf.nn.relu,use_bias=True)

            KW_transpose = tf.transpose(KW,[0,2,1])

            att_loc = tf.matmul(QW, KW_transpose)
            att_scale = tf.nn.softplus(tf.layers.dense(att_loc,self.num_steps,activation=tf.nn.leaky_relu,use_bias=True))

            att = tf.nn.softplus(tf.distributions.Normal(att_loc,att_scale).sample())

            att = att * helper

            self.att_each[to_task,from_task].append(att)

        with tf.variable_scope('transfer_from%d'%(from_task),reuse=tf.AUTO_REUSE):
            transformed_V = tf.layers.dense(V,self.num_hidden,activation=tf.nn.leaky_relu,use_bias=True,name="layer1")
            transformed_V = transformed_V / tf.norm(transformed_V,axis=2,keepdims=True)
            transfer_amount = tf.matmul(att,transformed_V)
        with tf.variable_scope('transfer_to%d'%(to_task),reuse=tf.AUTO_REUSE):
            transfer_amount = tf.layers.dense(transfer_amount,self.num_hidden,activation=tf.nn.leaky_relu,use_bias=True,name="layer1")
        return transfer_amount


    def build_model(self, use_lstm=True):
        print('Start building model')
        self.loss_each = []
        self.preds_each = []
        self.beta_att_each = []
        self.beta_main_each = []
        loss_task = 0
        self.att_each = {}
        self.test = []
        self.KL=[[0 for _ in range(self.num_tasks)] for i in range(self.num_tasks)]
        self.att_loc = {}
        self.att_scale = {}
        self.att_each = {(i,j):[] for i in range(self.num_tasks) for j in range(self.num_tasks)}
        self.att_each_before = {}
        self.z_in = {}
        self.test = {}

        self.beta_output = []

        self.helper = tf.contrib.distributions.fill_triangular(tf.ones([int(self.num_steps*(self.num_steps+1))/2]))
        with tf.variable_scope("base"):
            with tf.variable_scope("embed"):
                embed = tf.layers.dense(self.x,self.num_hidden,activation=None,use_bias=False)
                
            with tf.variable_scope("rnn"):
                def single_cell():    
                    lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
                    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, 
                                                         output_keep_prob=self.keep_prob,
                                                         variational_recurrent=True,
                                                         dtype=tf.float32
                                                         )   

                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_shared_rnn_layers)])
                r, _ = tf.nn.dynamic_rnn(cell,
                                        embed,
                                        dtype=tf.float32)

            for task_id in range(self.num_tasks):
                f = tf.reshape(r,[-1,self.num_hidden]) 
                with tf.variable_scope('task_%d/ffw'%(task_id)):
                    for i in range(self.num_layers-1):      
                        
                        W = tf.get_variable('weight_'+str(i),[self.num_hidden,self.num_hidden])
                        B = tf.get_variable('bias_'+str(i),[self.num_hidden])
                        W = tf.nn.dropout(W,keep_prob=self.keep_prob)     
                        f = tf.nn.leaky_relu(tf.matmul(f,W),B)
                        
                    W_loc1 = tf.get_variable('weight_loc1',[self.num_hidden,self.num_hidden])
                    B_loc1 = tf.get_variable('bias_loc1',[self.num_hidden])
                    W_loc1 = tf.nn.dropout(W_loc1,keep_prob=self.keep_prob)     
                    beta_loc = tf.nn.leaky_relu(tf.matmul(f,W_loc1),B_loc1)

                    W_loc2 = tf.get_variable('weight_loc2',[self.num_hidden,self.num_hidden])
                    B_loc2 = tf.get_variable('bias_loc2',[self.num_hidden])
                    W_loc2 = tf.nn.dropout(W_loc2,keep_prob=self.keep_prob)     
                    beta_loc = tf.nn.leaky_relu(tf.matmul(beta_loc,W_loc2),B_loc2)

                    beta_loc = tf.reshape(beta_loc,[-1,self.num_steps,self.num_hidden])

                    W_scale1 = tf.get_variable('weight_scale1',[self.num_hidden,self.num_hidden])
                    B_scale1 = tf.get_variable('bias_scale1',[self.num_hidden])
                    W_scale1 = tf.nn.dropout(W_scale1,keep_prob=self.keep_prob)     
                    beta_scale = tf.nn.leaky_relu(tf.matmul(f,W_scale1),B_scale1)
                    W_scale2 = tf.get_variable('weight_scale2',[self.num_hidden,self.num_hidden])
                    B_scale2 = tf.get_variable('bias_scale2',[self.num_hidden])
                    W_scale2 = tf.nn.dropout(W_scale2,keep_prob=self.keep_prob)     
                    beta_scale = tf.nn.leaky_relu(tf.matmul(beta_scale,W_scale2),B_scale2)
                    beta_scale = tf.reshape(beta_scale,[-1,self.num_steps,self.num_hidden])

                    beta_output = tf.distributions.Normal(beta_loc,beta_scale).sample()
                    

                self.beta_output.append(beta_output)

        self.beta_output_transfer = {}

        for task_id in range(self.num_tasks):
            beta_output_comb = self.beta_output[task_id]

            for source_id in range(self.num_tasks):
                if source_id != task_id:
                    source_knowledge = tf.stop_gradient(self.beta_output[source_id])
                else:
                    source_knowledge = self.beta_output[source_id]
                beta_output_transfer = self.transfer(self.beta_output[task_id],source_knowledge,source_knowledge,task_id,source_id)
                beta_output_comb = beta_output_comb + beta_output_transfer
                self.beta_output_transfer[source_id,task_id] = beta_output_transfer


            # attention_sum 
            logits = self.output(task_id, embed, beta_output_comb)
            preds = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[:,task_id:task_id+1])) 
            self.loss_each.append(loss)
            self.preds_each.append(preds)

            loss_task += loss 

        l2_losses = [tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='base') if ('kernel' in v.name or 'weight' in v.name)]
        loss_l2 = self.l2_coeff*tf.add_n(l2_losses)


        self.loss_sum = loss_task + loss_l2
        self.loss_all = {'loss_task':loss_task, 'loss_l2': loss_l2}
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss_sum)
        print ('Model built')

