import importlib
import argparse
from config import config
import os 
import pdb
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm
import pickle



# Load task-related code
task_module = importlib.import_module("task_codes." + config.task_code)
# Load task data and convert to pytorch
train_x_numpy , train_y_numpy, valid_x_numpy , valid_y_numpy, test_x_numpy , test_y_numpy, num_features, num_steps, num_tasks\
                 = task_module.load_data(tasks=config.tasks)

seq_len = train_x_numpy.shape[1]
input_features = train_x_numpy.shape[2]
# config.KL_scale = len(train_x_numpy)

# config.KL_scale = len(train_x_numpy)

train_x = torch.from_numpy(train_x_numpy).type(torch.FloatTensor) 
train_y = torch.from_numpy(train_y_numpy).type(torch.FloatTensor)
valid_x = torch.from_numpy(valid_x_numpy).type(torch.FloatTensor)
valid_y = torch.from_numpy(valid_y_numpy).type(torch.FloatTensor) 
test_x = torch.from_numpy(test_x_numpy).type(torch.FloatTensor) 
test_y = torch.from_numpy(test_y_numpy).type(torch.FloatTensor)

# config.TOTAL_EPOCH = int(100000/(len(train_x)/config.BATCH_SIZE))

def make_dataloader():
    

    # modify task config accordingly
    config.num_features = num_features
    config.num_steps = num_steps
    config.num_tasks = num_tasks


    datasets = {}
    datasets["train"] = torch.utils.data.TensorDataset(train_x,train_y) 
    datasets["valid"] = torch.utils.data.TensorDataset(valid_x,valid_y)
    datasets["test"] = torch.utils.data.TensorDataset(test_x,test_y) 


    dataloader = {}
    dataloader["train"] = torch.utils.data.DataLoader(datasets["train"], batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader["valid"] = torch.utils.data.DataLoader(datasets["valid"], batch_size=config.BATCH_SIZE, shuffle=False) 
    dataloader["test"] = torch.utils.data.DataLoader(datasets["test"], batch_size=config.BATCH_SIZE, shuffle=False)
    return dataloader


device = 'cuda'

dataloader = make_dataloader()

from adacare import *
# Load multi task class
net = AdaCare(hidden_dim=config.num_hidden,
                input_dim=config.num_features,
                output_dim=config.num_tasks)
net.to(device)

optimizer = optim.Adam(net.parameters(),config.lr)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.decay_rate)



save_path = "saved/%s_%s/"%(config.mtl_model,config.task_code)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
#saver.restore(sess, SAVE_DIR+"retain_mimic1400.ckpt")

def train_epoch():
    print("Training SAnD for task %s"%\
                        (config.task_code))
    total_loss = 0
    for batch_data, batch_labels in tqdm(dataloader["train"],ncols=75):
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        output, _ = net(batch_data,device)
        pred = output[:,-1,:]
        loss = F.binary_cross_entropy(pred,batch_labels,reduction='none')
        loss = loss.sum(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    total_loss = total_loss/len(dataloader["train"])
    print ('total_loss', total_loss)


def valid_epoch():
    print("--------------------------------------------------------")
    print("Performance on valid set")
    total_loss = 0
    total_pred = []
    for batch_data, batch_labels in tqdm(dataloader["valid"],ncols=75):
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        output, _ = net(batch_data,device)
        pred = output[:,-1,:]
        loss = F.binary_cross_entropy(pred,batch_labels,reduction='none')
        loss = loss.sum(1).mean()

        total_loss = total_loss + loss.item()
        total_pred.append(pred.to('cpu').data.numpy())

    total_loss = total_loss/len(dataloader["valid"])
    total_pred = np.concatenate(total_pred,0)
    auc = roc_auc_score(valid_y_numpy,total_pred,average=None)
    print ('loss', total_loss, 'auc', auc)

    return total_loss, auc



def train(e=0):
    # start training
    eval_loss_min = float('inf')
    eval_total_auc_best = [0 for _ in range(num_tasks)]
    eval_auc_best_for_each = [0 for _ in range(num_tasks)]
    epoch_min = 0
    best_model_filename = None

    try:
        for epoch in range(e,config.TOTAL_EPOCH):
            print("==========================================================")
            print(datetime.now(), best_model_filename)
            print ("Epoch: ",epoch+1)
            train_epoch()
            eval_loss, eval_auc = valid_epoch()
            if eval_loss<eval_loss_min:
                eval_loss_min = eval_loss
                eval_total_auc_best = eval_auc 
                epoch_min = epoch+1

                best_model_filename = save_path+'%d_%.3f'%(epoch+1,eval_loss)
                torch.save(net.state_dict(), best_model_filename)


    except KeyboardInterrupt:
        print()
    finally:
        print("******RESULT******")
        print("Valid loss min: %f at epoch %d. AUC is:, "%(eval_loss_min,epoch_min), eval_total_auc_best)

    return best_model_filename

def inference():
    print("==========================================================")
    print("==========================================================")
    print("==========================================================")
    print("Performance of the optimal model on test set")
    total_loss = 0
    total_pred = []
    for batch_data, batch_labels in tqdm(dataloader["test"],ncols=75):
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        output, _ = net(batch_data,device)
        pred = output[:,-1,:]
        loss = F.binary_cross_entropy(pred,batch_labels,reduction='none')
        loss = loss.sum(1).mean()

        total_loss = total_loss + loss.item()
        total_pred.append(pred.to('cpu').data.numpy())

    total_loss = total_loss/len(dataloader["valid"])
    total_pred = np.concatenate(total_pred,0)
    auc = roc_auc_score(test_y_numpy,total_pred,average=None)
    print ('loss', total_loss, 'auc', auc)




if __name__=="__main__":
    starting_epoch = 0
    # saved_model = 'saved/amtl_rnn_prob_mimic_infection_limit/175_2.069.ckpt'
    # print(saved_model)
    # starting_epoch = int(saved_model.split('/')[2].split('_')[0])
    # saver.restore(sess, saved_model)
    

    saved_model = train(0)
    net.load_state_dict(torch.load(saved_model))
    valid_epoch()
    inference()
