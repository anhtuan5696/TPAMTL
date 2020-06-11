import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
from collections import OrderedDict
import importlib
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


DATASET = 'mnist_rotation_back'
MODEL_NAME = 'amtl_loss'
MODE = 'inference'                   # training or inference

NUM_ITERS = 50000
CHECK_ITERS = 100
BATCH_SIZE = 64
LR = 1e-2
NUM_SAMPLING_INFERENCE = 5

# Model config
config = lambda: None

config.hidden_dim = 16
config.num_layers = 1


if DATASET == 'mnist_rotation_back':
    config.num_tasks = 2
    training_size = [2000,200]
    batch_size = [50,50]
    from load_mnist_variant import load
    data, targets =\
    load('MNIST_rotation_back/mnist_all_background_images_rotation_normalized_train_valid.amat')
    targets = targets.view([-1,1])
    targets_onehot = torch.FloatTensor(targets.shape[0],10).zero_()
    targets_onehot = targets_onehot.scatter_(1,targets,1)
    train_loaders = []
    train_sets = []
    for i in range(config.num_tasks):
        train_set = TensorDataset(data[:training_size[i]],targets_onehot[:training_size[i],i:i+1])
        train_sets.append(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size[i], shuffle=True, num_workers=1)
        train_loaders.append(train_loader)
    
    data, targets =\
    load('MNIST_rotation_back/mnist_all_background_images_rotation_normalized_test.amat')
    targets = targets.view([-1,1])
    targets_onehot = torch.FloatTensor(targets.shape[0],10).zero_()
    targets_onehot = targets_onehot.scatter_(1,targets,1)
    test_loaders = []
    for i in range(config.num_tasks):
        test_set = TensorDataset(data[:5000],targets_onehot[:5000,i:i+1])
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        test_loaders.append(test_loader)
    config.input_dim = 784

device = 'cuda'

# Define the model
model_class = getattr(importlib.import_module('models.'+MODEL_NAME), 'DataParallel_Model')
model = model_class(config)
model.to(device)

model_name = '%s_%s'%(MODEL_NAME,DATASET)


# Set up the optimizer and train step
optimizer = optim.SGD(model.parameters(),LR)

def train():
    model.train()
    epoch_pred_dict = [OrderedDict() for _ in range(config.num_tasks)]
    epoch_loss_dict = OrderedDict()
    epoch_auc_dict = [OrderedDict() for _ in range(config.num_tasks)]
    epoch_target = [[] for _ in range(config.num_tasks)]
    for i in tqdm(range(CHECK_ITERS),ncols=75,leave=False):
        xs = []
        ys = []
        for task_id in range(config.num_tasks):
            x,y = next(iter(train_loaders[task_id]))
            x = x.to(device)
            y = y.to(device)
            xs.append(x)
            ys.append(y)
            epoch_target[task_id].append(y.to('cpu').data.numpy())
        optimizer.zero_grad()

        main_loss, loss_dict, pred_dict = model(xs,ys)
        main_loss.backward()
        optimizer.step()
        
        for key in loss_dict.keys():
            if key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key].to('cpu').item()
            else:
                epoch_loss_dict[key] = 0
        for task_id in range(config.num_tasks):
            for key in pred_dict[task_id].keys():
                if key in epoch_pred_dict[task_id]:
                    epoch_pred_dict[task_id][key].append(pred_dict[task_id][key].to('cpu').data.numpy())
                else:
                    epoch_pred_dict[task_id][key] = [pred_dict[task_id][key].to('cpu').data.numpy()]
    for task_id in range(config.num_tasks):
        epoch_target[task_id] = np.concatenate(epoch_target[task_id],0)
    for key in epoch_loss_dict.keys():
        epoch_loss_dict[key] /= CHECK_ITERS
    for task_id in range(config.num_tasks):
        for key in epoch_pred_dict[task_id].keys():
            epoch_pred_dict[task_id][key] = np.concatenate(epoch_pred_dict[task_id][key],0)
            epoch_auc_dict[task_id][key] = roc_auc_score(epoch_target[task_id], epoch_pred_dict[task_id][key], average=None)
    return epoch_loss_dict, epoch_auc_dict

best_loss = float('inf')

def test_epoch():
    model.eval()
    epoch_pred_dict = [OrderedDict() for _ in range(config.num_tasks)]
    epoch_loss_dict = OrderedDict()
    epoch_auc_dict = [OrderedDict() for _ in range(config.num_tasks)]
    epoch_target = [[] for _ in range(config.num_tasks)]
    loss = 0.0
    for i in tqdm(range(len(test_loaders[0])),ncols=75,leave=False):
        xs = []
        ys = []
        for task_id in range(config.num_tasks):
            x,y = next(iter(test_loaders[task_id]))
            x = x.to(device)
            y = y.to(device)
            xs.append(x)
            ys.append(y)
            epoch_target[task_id].append(y.to('cpu').data.numpy())
        
        for s in range(NUM_SAMPLING_INFERENCE):
            main_l, l_dict, p_dict = model(xs,ys)
            if s == 0:
                main_loss = main_l
                loss_dict = l_dict
                pred_dict = p_dict
            else:
                main_loss = main_loss + main_l
                for key in loss_dict:
                    loss_dict[key] = loss_dict[key]+l_dict[key]
                for task_id in range(config.num_tasks):
                    for key in pred_dict[task_id]:
                        pred_dict[task_id][key] = pred_dict[task_id][key]+p_dict[task_id][key]
        main_loss = main_loss/NUM_SAMPLING_INFERENCE
        loss = loss + main_loss.to('cpu').item()
        for key in loss_dict:
            loss_dict[key] = loss_dict[key]/NUM_SAMPLING_INFERENCE
        for task_id in range(config.num_tasks):
            for key in pred_dict[task_id]:
                pred_dict[task_id][key] = pred_dict[task_id][key]/NUM_SAMPLING_INFERENCE
        
        for key in loss_dict.keys():
            if key in epoch_loss_dict:
                epoch_loss_dict[key] += loss_dict[key].to('cpu').item()
            else:
                epoch_loss_dict[key] = 0
        for task_id in range(config.num_tasks):
            for key in pred_dict[task_id].keys():
                if key in epoch_pred_dict[task_id]:
                    epoch_pred_dict[task_id][key].append(pred_dict[task_id][key].to('cpu').data.numpy())
                else:
                    epoch_pred_dict[task_id][key] = [pred_dict[task_id][key].to('cpu').data.numpy()]
    loss = loss/len(test_loaders[0])
    for task_id in range(config.num_tasks):
        epoch_target[task_id] = np.concatenate(epoch_target[task_id],0)
    for key in epoch_loss_dict.keys():
        epoch_loss_dict[key] /= len(test_loaders[0])
    for task_id in range(config.num_tasks):
        for key in epoch_pred_dict[task_id].keys():
            epoch_pred_dict[task_id][key] = np.concatenate(epoch_pred_dict[task_id][key],0)
            epoch_auc_dict[task_id][key] = roc_auc_score(epoch_target[task_id], epoch_pred_dict[task_id][key], average=None)
    return loss, epoch_loss_dict, epoch_auc_dict

if MODE=='training':
    print('Training')
    try:
        for i in range(int(NUM_ITERS/CHECK_ITERS)):
            print('%d. Train for %d iters:'%(i+1,CHECK_ITERS))
            print('\tTrain:')
            train_loss_dict, train_auc_dict = train()
            for key in train_loss_dict.keys():
                print('\t\t',key, train_loss_dict[key])
            for task_id in range(config.num_tasks):
                for key in train_auc_dict[task_id].keys():
                    print('\t\t Task %d AUC '%(task_id),key,train_auc_dict[task_id][key])


            print('\tTest:')
            loss, test_loss_dict, test_auc_dict = test_epoch()
            if loss<best_loss:
                best_loss = loss
                best_test_loss_dict = test_loss_dict
                best_test_auc_dict = test_auc_dict
                torch.save(model.state_dict(),'saved/%s_%s'%(MODEL_NAME,DATASET))
            torch.save(model.state_dict(),'saved/%s_%s_%d'%(MODEL_NAME,DATASET,i+1))
            for key in test_loss_dict.keys():
                print('\t\t',key, test_loss_dict[key])
            for task_id in range(config.num_tasks):
                for key in test_auc_dict[task_id].keys():
                    print('\t\t Task %d AUC '%(task_id),key,test_auc_dict[task_id][key])
    except Exception:
        pass
    finally:
        print()
        print()
        print('Best result')
        for key in best_test_loss_dict.keys():
            print('\t\t',key, best_test_loss_dict[key])
        for task_id in range(config.num_tasks):
            for key in best_test_auc_dict[task_id].keys():
                print('\t\t Task %d AUC '%(task_id),key,best_test_auc_dict[task_id][key])
if MODE=='inference':
    model.load_state_dict(torch.load('saved/amtl_loss_mnist_rotation_back_90'))
    test_loaders = []
    for i in range(config.num_tasks):
        loader = DataLoader(train_sets[i], batch_size=batch_size[i], shuffle=False, num_workers=1)
        test_loaders.append(loader)

    def infer_train_set():
        model.eval()
        epoch_pred_dict = [OrderedDict() for _ in range(config.num_tasks)]
        epoch_loss_dict = OrderedDict()
        epoch_auc_dict = [OrderedDict() for _ in range(config.num_tasks)]
        epoch_target = [[] for _ in range(config.num_tasks)]
        loss = 0.0
        for i in tqdm(range(len(test_loaders[0])),ncols=75,leave=False):
            xs = []
            ys = []
            for task_id in range(config.num_tasks):
                x,y = next(iter(test_loaders[task_id]))
                x = x.to(device)
                y = y.to(device)
                xs.append(x)
                ys.append(y)
                epoch_target[task_id].append(y.to('cpu').data.numpy())
            
            for s in range(NUM_SAMPLING_INFERENCE):
                main_l, l_dict, p_dict = model(xs,ys)
                if s == 0:
                    main_loss = main_l
                    loss_dict = l_dict
                    pred_dict = p_dict
                else:
                    main_loss = main_loss + main_l
                    for key in loss_dict:
                        loss_dict[key] = loss_dict[key]+l_dict[key]
                    for task_id in range(config.num_tasks):
                        for key in pred_dict[task_id]:
                            pred_dict[task_id][key] = pred_dict[task_id][key]+p_dict[task_id][key]
            main_loss = main_loss/NUM_SAMPLING_INFERENCE
            loss = loss + main_loss.to('cpu').item()
            for key in loss_dict:
                loss_dict[key] = loss_dict[key]/NUM_SAMPLING_INFERENCE
            for task_id in range(config.num_tasks):
                for key in pred_dict[task_id]:
                    pred_dict[task_id][key] = pred_dict[task_id][key]/NUM_SAMPLING_INFERENCE
            
            for key in loss_dict.keys():
                if key in epoch_loss_dict:
                    epoch_loss_dict[key] += loss_dict[key].to('cpu').item()
                else:
                    epoch_loss_dict[key] = 0
            for task_id in range(config.num_tasks):
                for key in pred_dict[task_id].keys():
                    if key in epoch_pred_dict[task_id]:
                        epoch_pred_dict[task_id][key].append(pred_dict[task_id][key].to('cpu').data.numpy())
                    else:
                        epoch_pred_dict[task_id][key] = [pred_dict[task_id][key].to('cpu').data.numpy()]
        loss = loss/len(test_loaders[0])
        for task_id in range(config.num_tasks):
            epoch_target[task_id] = np.concatenate(epoch_target[task_id],0)
        for key in epoch_loss_dict.keys():
            epoch_loss_dict[key] /= len(test_loaders[0])
        for task_id in range(config.num_tasks):
            for key in epoch_pred_dict[task_id].keys():
                epoch_pred_dict[task_id][key] = np.concatenate(epoch_pred_dict[task_id][key],0)
                epoch_auc_dict[task_id][key] = roc_auc_score(epoch_target[task_id], epoch_pred_dict[task_id][key], average=None)
        return loss, epoch_loss_dict, epoch_auc_dict

    print('Inference:')
    loss, test_loss_dict, test_auc_dict = infer_train_set()
    if loss<best_loss:
        best_loss = loss
        best_test_loss_dict = test_loss_dict
        best_test_auc_dict = test_auc_dict
        torch.save(model.state_dict(),'saved/%s_%s'%(MODEL_NAME,DATASET))
    torch.save(model.state_dict(),'saved/%s_%s_%d'%(MODEL_NAME,DATASET,i+1))
    for key in test_loss_dict.keys():
        print('\t\t',key, test_loss_dict[key])
    for task_id in range(config.num_tasks):
        for key in test_auc_dict[task_id].keys():
            print('\t\t Task %d AUC '%(task_id),key,test_auc_dict[task_id][key])


