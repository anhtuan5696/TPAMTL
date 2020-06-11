import numpy as np
import pdb

# data loader
path = 'physionet2012/'
def load_data(tasks=None):
    if tasks == None:
        task_set = [0,1,2,3]
    else:
        task_set = tasks
    # Get data
    data = np.load(path+'x.npy')
    label = np.load(path+'y.npy')

    label = label[:,task_set]

    inp = np.array(data)
    label = np.array(label)

    index = range(len(label))
    # np.random.shuffle(index)
    train = int(len(label)*0.7)
    valid = int(len(label)*0.8) 

    train_x = []
    train_y = []
    for i in range(train):
        train_x.append(inp[index[i]])
        train_y.append(label[index[i]])
    train_x = np.array(train_x,np.float32)
    train_y = np.array(train_y,np.float32)

    valid_x = []
    valid_y = []
    for i in range(train,valid):
        valid_x.append(inp[index[i]])
        valid_y.append(label[index[i]])
    valid_x = np.array(valid_x,np.float32)
    valid_y = np.array(valid_y,np.float32)

    eval_x = []
    eval_y = []
    for i in range(valid,len(index)):
        eval_x.append(inp[index[i]])
        eval_y.append(label[index[i]])
    eval_x = np.array(eval_x,np.float32)
    eval_y = np.array(eval_y,np.float32)


    return train_x, train_y, valid_x, valid_y, eval_x, eval_y, train_x.shape[2], train_x.shape[1], len(task_set)

