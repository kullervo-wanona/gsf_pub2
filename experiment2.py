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
from GenerativeSchurFlow2 import GenerativeSchurFlow

# from DataLoaders.MNIST.MNISTLoader import DataLoader
# from DataLoaders.MNIST.ColorMNISTLoader import DataLoader
# from DataLoaders.CelebA.CelebA32Loader import DataLoader
from DataLoaders.CelebA.CelebA64Loader import DataLoader
# from DataLoaders.CIFAR.Cifar10 import DataLoader

train_data_loader = DataLoader(batch_size=20)
train_data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(train_data_loader) 

test_data_loader = DataLoader(batch_size=20)
test_data_loader.setup('Test', randomized=False, verbose=False)
_, _, example_test_batch = next(test_data_loader) 
test_image = helper.cuda(torch.from_numpy(example_test_batch['Image']))

c_in=train_data_loader.image_size[1]
n_in=train_data_loader.image_size[3]

# flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[3]*10, squeeze_list=[0]*10)
flow_net = GenerativeSchurFlow(c_in, n_in, k_list=[5]*5+[3]*5, squeeze_list=([0]*2+[1]*1+[0]*2+[1]*1+[0]*4))
flow_net.set_actnorm_parameters(train_data_loader, setup_mode='Training', n_batches=5, test_normalization=False)

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.is_cuda, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

n_param = 0
for e in flow_net.parameters():
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

# optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-08)
optimizer = torch.optim.Adam(flow_net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)

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
        torch.nn.utils.clip_grad_norm_(flow_net.parameters(), 0.4)
        # torch.nn.utils.clip_grad_norm_(flow_net.parameters(), 0.1) # worked
        optimizer.step()

        # if i % 20 == 0:
        if i % 400 == 0:
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

            # if i % 1000 == 0:
            #     test_all_z = flow_net.transform_all_layers(test_image)
            #     for layer_id in range(len(test_all_z)):
            #         helper.vis_samples_np(helper.cpu(test_all_z[layer_id]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/layers/layer_'+str(layer_id)+'_test/', prefix='real', resize=[256, 256])
                
            #     test_rec = flow_net.inverse_transform_all_layers(test_all_z[-1])
            #     for layer_id in range(len(test_rec)):
            #         helper.vis_samples_np(helper.cpu(test_rec[layer_id]).detach().numpy(), sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_schur/inverse_layers/inverse_layer_'+str(layer_id)+'_test/', prefix='real', resize=[256, 256])

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




















