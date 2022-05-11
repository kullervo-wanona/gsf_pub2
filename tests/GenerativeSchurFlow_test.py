import os, sys, inspect
sys.path.insert(1, os.path.realpath(os.path.pardir))

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

from DataLoaders.CelebA.CelebA32Loader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(data_loader) 

c_in = 3
n_in = 16
flow_net = GenerativeSchurFlow(c_in=c_in, n_in=n_in, k_list=[3, 4, 4], squeeze_list=[0, 1, 0])
flow_net.set_actnorm_parameters(data_loader, setup_mode='Training', n_batches=10, 
    test_normalization=True, sub_image=[c_in, n_in, n_in])

n_param = 0
for name, e in flow_net.named_parameters():
    print(name, e.requires_grad, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

example_input = helper.cuda(torch.from_numpy(example_batch['Image']))[:, :c_in, :n_in, :n_in]
example_input_np = helper.to_numpy(example_input)

example_out, logdet_computed = flow_net.transform_with_logdet(example_input)
example_input_reconst = flow_net.inverse_transform(example_out) 
example_input_reconst_np = helper.to_numpy(example_input_reconst)

inversion_error_max = np.abs(example_input_reconst_np-example_input_np).max()
inversion_error_mean = np.abs(example_input_reconst_np-example_input_np).mean()
inversion_error_median = np.median(np.abs(example_input_reconst_np-example_input_np))
print('Inversion error max:' + str(inversion_error_max))
print('Inversion error mean:' + str(inversion_error_mean))
print('Inversion error median:' + str(inversion_error_median))

z, x, logdet, log_pdf_z, log_pdf_x = flow_net(example_input)
x_sample = flow_net.sample_x(n_samples=10)

J, J_flat = flow_net.jacobian(example_input)
det_sign, logdet_desired_np = np.linalg.slogdet(J_flat)

logdet_computed_np = helper.to_numpy(logdet_computed)
logdet_desired_error = np.abs(logdet_desired_np-logdet_computed_np).max()
print("Desired Logdet: \n", logdet_desired_np)
print("Computed Logdet: \n", logdet_computed_np)
print('Logdet error:' + str(logdet_desired_error))

print('Inversion error max:' + str(inversion_error_max))
print('Inversion error mean:' + str(inversion_error_mean))
print('Inversion error median:' + str(inversion_error_median))

assert (logdet_desired_error < 1e-3)
assert (inversion_error_max < 1e-3)
print('All tests passed.')








































