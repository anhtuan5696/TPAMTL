import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import scipy
import os
import seaborn as sns

#dataset = 'mimic-infection'
#path = '/st2/tuan_home/AMTL/images_mimic/'

dataset = 'physionet'
path = './images_physionet/'

if dataset == 'physionet':    
    num_steps = 49
    num_features = 8
    num_samples = 30
    num_tasks = 4
    task = ['Mortality','Stay<3','Cardiac','Recovery']
    
elif dataset == 'mimic-infection':
    num_steps = 48
    num_features = 8
    num_samples = 30
    num_tasks = 4
    task = ['Fever','Leukocytosis','Infection','Mortality']

beta_outputs = pickle.load(open(path+'beta_outputs.pkl','rb'))
att_eachs = pickle.load(open(path+'att_eachs.pkl','rb'))

beta_outputs_task = [None for _ in range(4)]

for i in range(4):
    beta_outputs_task[i] = [x[i] for x in beta_outputs] 


pid = 1



folder = path + 'customm/%d/'%(pid)

# amount-source
source_id = 2
target_id = 0
steps = range(0,23)
w_i = 0
w_s = None

source_avr_vars = []
source_avr_means = []
for step in steps:
    total_var = 0
    total_mean = 0
    for f in range(num_features):
        total_var += np.var([beta_outputs_task[source_id][sam][pid][step][f] for sam in range(num_samples)])
        total_mean += np.mean([beta_outputs_task[source_id][sam][pid][step][f] for sam in range(num_samples)])
    avr_var = total_var / num_features
    avr_mean = total_mean / num_features
    source_avr_vars.append(avr_var)
    source_avr_means.append(avr_mean)

transfer_total = []
for step in steps:
    transfers = []
    transfer_stds = []
    target_step_max = num_steps if w_s==None else min(num_steps,step+w_s)
    for target_step in range(step,target_step_max):
        transfer = np.mean([att_each[target_id,source_id][w_i][pid][target_step,step] for att_each in att_eachs])
        transfers.append(transfer)
    transfers = np.mean(transfers)
    transfer_total.append(transfers)


fig,ax = plt.subplots(figsize=[10,3.5])
color = 'red'
ax = sns.lineplot(steps, source_avr_vars,marker='.',color=color,ax=ax)
ax.legend(loc='upper left')
ax.set_ylabel('Uncertainty of Source',fontsize=18)
ax.set_ylim(top=max(source_avr_vars)*1.1)
ax.margins(y=0)
ax.spines['right'].set_color(color)
ax.tick_params(axis='y', colors=color)
ax.yaxis.label.set_color(color)


ax1 = ax.twinx()
sns.lineplot(steps, transfer_total,marker='.', color='#2976b1',ax=ax1)
ma = max(transfer_total)
mi = min(transfer_total)
ax1.set_ylim([mi,ma])
ax1.set_ylabel('Transfer Amount',fontsize=18)
ax1.spines['right'].set_color('#2976b1')
ax1.tick_params(axis='y', colors='#2976b1')
ax1.yaxis.label.set_color('#2976b1')
# ax1.set_ylim([mi*3/4,ma*2])
ax1.set_xticks(steps)
ax1.legend(loc="upper right")
ax1.margins(y=0)
ax.set_title('Knowledge Transfer from %s to %s: \nUncertainty of Source features vs Normalized Total Knowledge transfer'%(task[source_id],task[target_id]),fontsize=18)

if not os.path.exists(folder+'/Transfer%s_%s/'%(task[source_id],task[target_id])):
    os.makedirs(folder+'/Transfer%s_%s/'%(task[source_id],task[target_id]))
ax = fig.get_figure()
fig.savefig(folder+'/Transfer%s_%s/source_amount.png'%(task[source_id],task[target_id]),bbox_inches='tight')
plt.close(fig)



# amount-target
source_id = 0
target_id = 2
steps = range(15,38)
w_i = 0
w_s = None

transfer_total = []
for step in steps:
    transfers = []
    source_step_min = 0 if w_s==None else max(0,step-w_s+1)
    for source_step in range(source_step_min,step+1):
        transfer = np.mean([att_each[target_id,source_id][w_i][pid][step,source_step] for att_each in att_eachs])
        transfers.append(transfer)
    transfers = np.mean(transfers)
    transfer_total.append(transfers)

target_avr_vars = []
target_avr_means = []

for step in steps:
    total_var = 0
    total_mean = 0
    for f in range(num_features):
        total_var += np.var([beta_outputs_task[target_id][sam][pid][step][f] for sam in range(num_samples)])
        total_mean += np.square(np.mean([beta_outputs_task[target_id][sam][pid][step][f] for sam in range(num_samples)]))
    avr_var = total_var / num_features
    avr_mean = np.sqrt(total_mean / num_features)
    target_avr_vars.append(avr_var)
    target_avr_means.append(avr_mean)

fig,ax = plt.subplots(figsize=[10,3.5])
color = 'red'
ax = sns.lineplot(steps, target_avr_vars,marker='.',color=color,ax=ax)
ax.set_ylim(top=max(target_avr_vars)*1.1)
ax.set_ylabel('Uncertainty of Target',fontsize=18)
ax.legend(loc='upper left')

ax.spines['right'].set_color(color)
ax.tick_params(axis='y', colors=color)
ax.yaxis.label.set_color(color)

ax1 = ax.twinx()
sns.lineplot(steps, transfer_total,marker='.', color='#2976b1',ax=ax1)
ma = max(transfer_total)
mi = min(transfer_total)
ax1.set_ylim(mi,ma)
ax1.set_ylabel('Transfer Amount',fontsize=18)
ax1.spines['right'].set_color('#2976b1')
ax1.tick_params(axis='y', colors='#2976b1')
ax1.yaxis.label.set_color('#2976b1')
# ax1.set_ylim([mi*3/4,ma*2])
ax1.set_xticks(steps)
ax1.legend(loc="upper right")
ax.set_title('Knowledge Transfer from %s to %s: \nUncertainty of Target Features vs Normalized Total Knowledge transfer'%(task[source_id],task[target_id]),fontsize=18)

if not os.path.exists(folder+'/Transfer%s_%s/'%(task[source_id],task[target_id])):
    os.makedirs(folder+'/Transfer%s_%s/'%(task[source_id],task[target_id]))
fig = ax.get_figure()
fig.savefig(folder+'/Transfer%s_%s/target_amount.png'%(task[source_id],task[target_id]),bbox_inches='tight')
plt.close(fig)
