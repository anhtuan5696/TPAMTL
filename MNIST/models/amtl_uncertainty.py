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
        for i in range(self.num_tasks):
            self.module_dict['base_task%d'%(i)] = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),nn.ReLU())
            self.module_dict['mu_task%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
            self.module_dict['sigma_task%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
            self.module_dict['outlayer_without_transfer_task%d'%(i)] = nn.Linear(self.hidden_dim,1)
            self.module_dict['outlayer_with_transfer_task%d'%(i)] = nn.Linear(self.hidden_dim,1)
            for j in range(self.num_tasks):
                self.module_dict['F_%d_%d'%(i,j)] = nn.Sequential(nn.Linear(4*self.hidden_dim,1),nn.Sigmoid())
            self.module_dict['G_encoder_%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
            self.module_dict['G_decoder_%d'%(i)] = nn.Linear(self.hidden_dim,self.hidden_dim)
    def forward(self,x,y):
        hidden = [None for _ in range(self.num_tasks)]
        mu = [None for _ in range(self.num_tasks)]
        sigma = [None for _ in range(self.num_tasks)]
        pred_with_transfer = [None for _ in range(self.num_tasks)]
        loss_with_transfer = [None for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            h = self.module_dict['base_task%d'%(i)](x[i])
            mu[i] = self.module_dict['mu_task%d'%(i)](h)
            sigma[i] = F.softplus(self.module_dict['sigma_task%d'%(i)](h))
            epsilon = torch.distributions.normal.Normal(0,1).sample([1,mu[i].shape[1]]).to(mu[i].device)
            hidden[i] = mu[i] + sigma[i]*epsilon
        
        for i in range(self.num_tasks):
            combine = hidden[i]
            for j in range(self.num_tasks):
                if i!=j:
                    borrow_h = self.module_dict['base_task%d'%(j)](x[i])
                    borrow_mu = self.module_dict['mu_task%d'%(j)](borrow_h)
                    borrow_sigma = F.softplus(self.module_dict['sigma_task%d'%(j)](borrow_h))
                    epsilon = torch.distributions.normal.Normal(0,1).sample([1,borrow_mu.shape[1]]).to(borrow_mu.device)
                    borrow_hidden = borrow_mu + borrow_sigma*epsilon
                    information_source = torch.cat([borrow_sigma,borrow_hidden],1).detach()
                    information_target = torch.cat([sigma[i],hidden[i]],1).detach()
                    transfer_weight = self.module_dict['F_%d_%d'%(i,j)](torch.cat([information_source,information_target],1))
                    pdb.set_trace()
                    combine = combine + self.module_dict['G_decoder_%d'%(i)](transfer_weight*self.module_dict['G_encoder_%d'%(j)](borrow_hidden.detach()))
                    #transfer_matrix[i,j] = transfer_weight[0][0]
            pred_with_transfer[i] = torch.sigmoid(self.module_dict['outlayer_with_transfer_task%d'%(i)](combine))
            loss_with_transfer[i] = F.binary_cross_entropy(pred_with_transfer[i], y[i], reduction='none')
        #print(transfer_matrix)
        #pred_without_transfer = torch.cat(pred_without_transfer,1)
        #pred_with_transfer = torch.cat(pred_with_transfer,1)
        return loss_with_transfer, pred_with_transfer

class DataParallel_Model(nn.Module):
    def __init__(self,config):
        super(DataParallel_Model, self).__init__()
        self.num_tasks = config.num_tasks
        self.model = Model(config)
        self.model = nn.DataParallel(self.model)
    def forward(self,x,y):
        loss_with_transfer, pred_with_transfer = self.model(x,y)
        loss_with_transfer = [loss_with_transfer[i].mean() for i in range(self.num_tasks)]
        main_loss = sum(loss_with_transfer)

        loss_dict = OrderedDict()
        loss_dict['Main Loss:'] = main_loss
        for i in range(self.num_tasks):
            loss_dict['Task %d Loss with Transfer:'%(i)] = loss_with_transfer[i]

        pred_dict = [OrderedDict() for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            pred_dict[i]['with Transfer:'] = pred_with_transfer[i]

        return main_loss, loss_dict, pred_dict

