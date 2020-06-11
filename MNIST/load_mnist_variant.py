import numpy as np
import torch
def load(file_name):
    """ Parsing & loading """
    with open(file_name, 'r') as f:
        data = f.read().split('\n')
    data = data[:-1]
    x = np.zeros((len(data), 784))
    y = np.zeros(len(data))
    for i in range(len(data)):
        xi = data[i].split()
        for j in range(784):
            x[i][j] = float(xi[j])
        y[i] = float(xi[784])
    return [torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)]
