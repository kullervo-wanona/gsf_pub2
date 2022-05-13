import os, sys, inspect

from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import time
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

import helper
from GenerativeSchurFlow import GenerativeSchurFlow
# from GenerativeConditionalSchurFlow import GenerativeConditionalSchurFlow
# from GenerativeConditionalSchurFlow2 import GenerativeConditionalSchurFlow
# from GenerativeConditionalSchurFlow3 import GenerativeConditionalSchurFlow
from GenerativeConditionalSchurFlow4 import GenerativeConditionalSchurFlow

# from DataLoaders.MNIST.MNISTLoader import DataLoader
from DataLoaders.MNIST.ColorMNISTLoader import DataLoader
# from DataLoaders.CelebA.CelebA32Loader import DataLoader
# from DataLoaders.CIFAR.Cifar10 import DataLoader
# trace()

train_data_loader = DataLoader(batch_size=20)
train_data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(train_data_loader) 

test_data_loader = DataLoader(batch_size=20)
test_data_loader.setup('Test', randomized=False, verbose=False)
_, _, example_test_batch = next(test_data_loader) 
test_image = helper.cuda(torch.from_numpy(example_test_batch['Image']))

c_in=train_data_loader.image_size[1]
n_in=train_data_loader.image_size[3]

# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[20, 20, 20], squeeze_list=[0, 0, 0])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[20, 10, 7, 7, 7, 7, 7], squeeze_list=[0, 1, 1, 0, 0, 0, 0])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[20, 10, 7], squeeze_list=[0, 1, 1])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[3, 4, 5, 6, 7])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[3, 3, 3, 3, 3, 3])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[20, 20, 20], squeeze_list=[0, 0, 0])

# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[10]*7, squeeze_list=[0]*7)
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[10]*10, squeeze_list=[0]*10)
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[7]*5+[5]*5+[3]*5, squeeze_list=[0]*5+[1]+[0]*4+[1]+[0]*4)
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10], squeeze_list=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[10, 10, 10, 10, 10], squeeze_list=[0, 0, 0, 0, 0])
# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10], squeeze_list=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
flow_net = GenerativeConditionalSchurFlow(c_in, n_in, n_blocks=10)
# flow_net.set_actnorm_parameters(train_data_loader, setup_mode='Training', n_batches=3, test_normalization=False)
flow_net.set_actnorm_parameters(train_data_loader, setup_mode='Training', n_batches=3, test_normalization=False)

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.is_cuda, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

n_param = 0
for e in flow_net.parameters():
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.9, 0.95), eps=1e-08)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-08, weight_decay=5e-5)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-08)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-08)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-08, weight_decay=5e-5)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0003, betas=(0.5, 0.9), eps=1e-08, weight_decay=5e-5)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.001, betas=(0.9, 0.9))
optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.001, betas=(0,  0.5), eps=1e-08)
# optimizer = torch.optim.RMSprop(flow_net.parameters(), lr=0.0001, alpha=0.9, eps=1e-08, weight_decay=0, momentum=0.5, centered=False)

exp_t_start = time.time()
for epoch in range(100000):
    train_data_loader.setup('Training', randomized=True, verbose=True)
    for i, curr_batch_size, batch_np in train_data_loader:     

        train_image = helper.cuda(torch.from_numpy(batch_np['Image']))
        optimizer.zero_grad() 

        z, x, logdet, log_pdf_z, log_pdf_x = flow_net(train_image)
        # train_loss = -torch.mean(logdet)-2*torch.mean(log_pdf_z)
        train_loss = -torch.mean(log_pdf_x)

        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(flow_net.parameters(), 0.25)
        # torch.nn.utils.clip_grad_norm_(flow_net.parameters(), 0.1) # worked
        optimizer.step()

        if i % 50 == 0:
        # if i % 200 == 0:
            train_latent, _ = flow_net.transform_with_logdet(train_image)
            train_image_reconst = flow_net.inverse_transform(train_latent)

            test_latent, _ = flow_net.transform_with_logdet(test_image)
            test_image_reconst = flow_net.inverse_transform(test_latent)

            _, _, _, train_log_pdf_z, train_log_pdf_x = flow_net(train_image)
            mean_train_log_pdf_z = torch.mean(train_log_pdf_z)
            mean_train_log_pdf_x = torch.mean(train_log_pdf_x)

            _, _, _, test_log_pdf_z, test_log_pdf_x = flow_net(train_image)
            mean_test_log_pdf_z = torch.mean(test_log_pdf_z)
            mean_test_log_pdf_x = torch.mean(test_log_pdf_x)

            image_sample = flow_net.sample_x(n_samples=50)            
            image_sharper_sample = flow_net.sample_sharper_x(n_samples=50)            

            helper.vis_samples_np(helper.cpu(train_image).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_real/', prefix='real', resize=[256, 256])
            helper.vis_samples_np(helper.cpu(train_image_reconst).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/train_reconst/', prefix='reconst', resize=[256, 256])

            helper.vis_samples_np(helper.cpu(test_image).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/test_real/', prefix='real', resize=[256, 256])
            helper.vis_samples_np(helper.cpu(test_image_reconst).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/test_reconst/', prefix='reconst', resize=[256, 256])

            helper.vis_samples_np(helper.cpu(image_sample).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/sample/', prefix='sample', resize=[256, 256])
            helper.vis_samples_np(helper.cpu(image_sharper_sample).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/sharper_sample/', prefix='sharper_sample', resize=[256, 256])

            train_log_likelihood_z = mean_train_log_pdf_z.item()
            train_log_likelihood_x = mean_train_log_pdf_x.item()
            train_nats_per_dim = train_log_likelihood_x/np.prod(train_image.shape[1:])
            train_bits_per_dim = train_nats_per_dim/np.log(2)

            test_log_likelihood_z = mean_test_log_pdf_z.item()
            test_log_likelihood_x = mean_test_log_pdf_x.item()
            test_nats_per_dim = test_log_likelihood_x/np.prod(test_image.shape[1:])
            test_bits_per_dim = test_nats_per_dim/np.log(2)

            print(f'[{epoch + 1}, {i + 1:5d}] Train Z LL, X LL, nats, bits: {train_log_likelihood_z, train_log_likelihood_x, train_nats_per_dim, train_bits_per_dim}')
            print(f'[{epoch + 1}, {i + 1:5d}] Test Z LL, X LL, nats, bits: {test_log_likelihood_z, test_log_likelihood_x, test_nats_per_dim, test_bits_per_dim}')

print('Experiment took '+str(time.time()-exp_t_start)+' seconds.')
print('Finished Training')






# log_2 x = log_e x * mult
# mult = log_2 x/log_e x = (log x/log 2)/(log x/log e) = 1/log 2






















# def leaky_relu_with_logdet(x, neg_slope=0.2):
#     x_pos = torch.nn.functional.relu(x)
#     x_neg = x-x_pos
#     y = x_pos+neg_slope*x_neg
#     y_deriv = helper.cuda(torch.ge(x, 0).type(torch.float32))
#     y_deriv += neg_slope*(1-y_deriv)
#     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
#     return y, y_logdet
    
# def sigmoid_with_logdet(x):
#     y = torch.sigmoid(x)
#     y_deriv = (1-y)*y
#     y_logdet = torch.log(y_deriv).sum(axis=[1, 2, 3])
#     return y, y_logdet

    