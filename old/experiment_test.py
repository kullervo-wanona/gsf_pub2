from sys import platform
if 'linux' in platform: 
    # import IPython.core.debugger
    # trace = IPython.core.debugger.Pdb.set_trace() #this one triggers the debugger
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import time
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch
import helper

import spectral_schur_det_lib
from multi_channel_invertible_conv_lib import spatial_conv2D_lib
from multi_channel_invertible_conv_lib import frequency_conv2D_lib

# torch.set_flush_denormal(True)

# from DataLoaders.CelebA.CelebA128Loader import DataLoader
# # from DataLoaders.CelebA.CelebA64Loader import DataLoader
# data_loader = DataLoader(batch_size=10)
# data_loader.setup('Training', randomized=True, verbose=False)
# data_loader.setup('Test', randomized=False, verbose=False)
# _, _, batch = next(data_loader)

from DataLoaders.MNIST.MNISTLoader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
# data_loader.setup('Test', randomized=True, verbose=True)
# data_loader.setup('Validation', randomized=True, verbose=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(3, 16, 5)
        self.fc1 = torch.nn.Linear(16*4*4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.nn.functional.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.nn.functional.relu(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        conv1_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 1, 50, 'he_uniform')
        self.conv1_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv1_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv1_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((50), dtype=torch.float32), requires_grad=True))

        conv2_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 50, 128, 'he_uniform')
        self.conv2_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv2_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv2_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((128), dtype=torch.float32), requires_grad=True))

        # self.fc1 = torch.nn.Linear(128*4*4, 120)
        # self.fc2 = torch.nn.Linear(120, 84)
        # self.fc3 = torch.nn.Linear(84, 10)
        # self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128*4*4, 120).to(device='cuda')
        self.fc2 = torch.nn.Linear(120, 84).to(device='cuda')
        self.fc3 = torch.nn.Linear(84, 10).to(device='cuda')
        self.pool = torch.nn.MaxPool2d(2, 2).to(device='cuda')

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.conv1_kernel, bias=self.conv1_bias, stride=(1, 1), padding='valid', dilation=(1, 1))
        x = self.pool(torch.nn.functional.relu(x))
        x = torch.nn.functional.conv2d(x, self.conv2_kernel, bias=self.conv2_bias, stride=(1, 1), padding='valid', dilation=(1, 1))
        x = self.pool(torch.nn.functional.relu(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        conv1_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 1, 50, 'he_uniform')
        self.conv1_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv1_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv1_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((50), dtype=torch.float32), requires_grad=True))

        conv2_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 50, 64, 'he_uniform')
        self.conv2_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv2_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv2_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((64), dtype=torch.float32), requires_grad=True))

        # self.fc1 = torch.nn.Linear(128*7*7, 120)
        # self.fc2 = torch.nn.Linear(120, 84)
        # self.fc3 = torch.nn.Linear(84, 10)
        # self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(64*7*7, 120).to(device='cuda')
        self.fc2 = torch.nn.Linear(120, 84).to(device='cuda')
        self.fc3 = torch.nn.Linear(84, 10).to(device='cuda')
        self.pool = torch.nn.MaxPool2d(2, 2).to(device='cuda')

    def forward(self, x):
        x = spatial_conv2D_lib.spatial_circular_conv2D_th(x, self.conv1_kernel, bias=self.conv1_bias)
        x = self.pool(torch.nn.functional.relu(x))
        x = spatial_conv2D_lib.spatial_circular_conv2D_th(x, self.conv2_kernel, bias=self.conv2_bias)
        x = self.pool(torch.nn.functional.relu(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class Net3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        conv1_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 1, 50, 'he_uniform')
        self.conv1_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv1_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv1_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((50), dtype=torch.float32), requires_grad=True))

        conv2_kernel_np = helper.get_conv_initial_weight_kernel_np([5, 5], 50, 64, 'he_uniform')
        self.conv2_kernel = helper.cuda(torch.nn.parameter.Parameter(data=torch.tensor(conv2_kernel_np, dtype=torch.float32), requires_grad=True))
        self.conv2_bias = helper.cuda(torch.nn.parameter.Parameter(data=torch.zeros((64), dtype=torch.float32), requires_grad=True))

        self.fc1 = torch.nn.Linear(128*7*7, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)

        # self.fc1 = torch.nn.Linear(64*7*7, 120).to(device='cuda')
        # self.fc2 = torch.nn.Linear(120, 84).to(device='cuda')
        # self.fc3 = torch.nn.Linear(84, 10).to(device='cuda')
        # self.pool = torch.nn.MaxPool2d(2, 2).to(device='cuda')

    def forward(self, x):
        x = spatial_conv2D_lib.spatial_circular_conv2D_th(x, self.conv1_kernel, bias=self.conv1_bias)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = spatial_conv2D_lib.spatial_circular_conv2D_th(x, self.conv2_kernel, bias=self.conv2_bias)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        
        return x   

# net = Net()
# net = Net2()
net = Net3()

criterion = torch.nn.CrossEntropyLoss()
for e in net.parameters(): print(e.shape)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.5)

exp_t_start = time.time()
running_loss = 0.0
for epoch in range(1):

    data_loader.setup('Training', randomized=True, verbose=True)
    for i, curr_batch_size, batch_np in data_loader:     
        image = helper.cuda(torch.from_numpy(batch_np['Image']))
        label = helper.cuda(torch.from_numpy(batch_np['Label']))

        optimizer.zero_grad() # zero the parameter gradients

        output = net(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 400 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss = 0.0

print('Experiment took '+str(time.time()-exp_t_start)+' seconds.')
print('Finished Training')


