import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from collections import OrderedDict
import numpy as np

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))
        
        self.module_dict = nn.ModuleDict()
        self.module_dict['base'] = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),nn.ReLU())
        self.module_dict['mu'] = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.module_dict['sigma'] = nn.Linear(self.hidden_dim,self.hidden_dim)
        for i in range(self.num_tasks):
            self.module_dict['outlayer_task%d'%(i)] = nn.Linear(self.hidden_dim,1)
            for j in range(self.num_tasks):
                self.module_dict['F_%d_%d'%(i,j)] = nn.Sequential(nn.Linear(2,1),nn.Sigmoid())
            self.module_dict['G_encoder_%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
            self.module_dict['G_decoder_%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
    def forward(self,x,y):
        hidden = [None for _ in range(self.num_tasks)]
        mu = [None for _ in range(self.num_tasks)]
        sigma = [None for _ in range(self.num_tasks)]
        pred = [None for _ in range(self.num_tasks)]
        loss = [None for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            h = self.module_dict['base'](x[i])
            mu[i] = self.module_dict['mu'](h)
            sigma[i] = F.softplus(self.module_dict['sigma'](h))
            epsilon = torch.distributions.normal.Normal(0,1).sample([1,mu[i].shape[1]]).to(mu[i].device)
            hidden[i] = mu[i] + sigma[i]*epsilon
            pred[i] = torch.sigmoid(self.module_dict['outlayer_task%d'%(i)](hidden[i]))
            loss[i] = F.binary_cross_entropy(pred[i], y[i], reduction='none')
        return loss, pred        

class DataParallel_Model(nn.Module):
    def __init__(self,config):
        super(DataParallel_Model, self).__init__()
        self.num_tasks = config.num_tasks
        self.model = Model(config)
        self.model = nn.DataParallel(self.model)
    def forward(self,x,y):
        loss, pred = self.model(x,y)
        loss = [loss[i].mean() for i in range(self.num_tasks)]
        main_loss = sum(loss)

        loss_dict = OrderedDict()
        loss_dict['Main Loss:'] = main_loss
        for i in range(self.num_tasks):
            loss_dict['Task %d Loss:'%(i)] = loss[i]

        pred_dict = [OrderedDict() for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            pred_dict[i]['MTL:'] = pred[i]

        return main_loss, loss_dict, pred_dict

