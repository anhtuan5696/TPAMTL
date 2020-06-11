# config
config = lambda: None

# Experiment setup
config.mtl_model = "tp_amtl_unconstrained"

#config.task_code = "mimic_infection"
config.task_code = "physionet2012"
#config.task_code = "mimic_heartfailure"

config.tasks = None
# for single-task version of UA,LSTM and RETAIN model, simply select config.tasks = [i] where i is the task you want to run (0,1,2,etc.)


# multitask model setup:
config.l2_coeff = 0.0002

config.asym_lambda = 0.01
config.asym_mu = 0.01                  # only used in AMTL Lee 2016

config.num_samples = 20

# singletask learner setup
config.num_hidden = 8
config.num_layers = 2

config.num_shared_rnn_layers = 1 #or 1

config.num_training_samples = 5

# training config
config.TOTAL_EPOCH = 2000
config.BATCH_SIZE = 128
config.lr = 1e-3
config.decay_rate = 0.1
config.decay_iter = 56000

