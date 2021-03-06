from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import numpy as np
import torch

import helper
from Transforms import MultiChannel2DCircularConv, AffineInterpolate, PReLU, SLogGate, Actnorm, Squeeze # Logit, Tanh, FixedSLogGate, , 

class GenerativeSchurFlow(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list, squeeze_list, final_actnorm=False):
        super().__init__()
        assert (len(k_list) == len(squeeze_list))
        self.name = 'GenerativeSchurFlow'
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.squeeze_list = squeeze_list
        self.final_actnorm = final_actnorm
        self.n_layers = len(self.k_list)

        self.uniform_dist = torch.distributions.Uniform(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_sharper_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.7])))

        print('\n**********************************************************')
        print('Creating GenerativeSchurFlow: ')
        print('**********************************************************\n')
        actnorm_layers, pre_interpolation_layers, pre_nonlin_layers = [], [], [] 
        conv_layers, interpolation_layers, nonlin_layers = [], [], []

        accum_squeeze = 0
        for layer_id in range(self.n_layers):
            accum_squeeze += self.squeeze_list[layer_id]
            curr_c = self.c_in*(4**accum_squeeze)
            curr_n = self.n_in//(2**accum_squeeze)
            curr_k = self.k_list[layer_id]
            print('Layer '+str(layer_id)+': c='+str(curr_c)+', n='+str(curr_n)+', k='+str(curr_k))
            assert (curr_n >= curr_k)

            actnorm_layers.append(Actnorm(curr_c, curr_n, name=str(layer_id)))

            pre_interpolation_layers.append(AffineInterpolate(curr_c, curr_n, name=str(layer_id)))
            
            # pre_nonlin_layers.append(SLogGate(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))
            # pre_nonlin_layers.append(PReLU(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))
            pre_nonlin_layers.append(PReLU(curr_c, curr_n, mode='spatial', name=str(layer_id)))

            conv_layers.append(MultiChannel2DCircularConv(
                curr_c, curr_n, curr_k, kernel_init='I + he_uniform', 
                bias_mode='spatial', scale_mode='no-scale', name=str(layer_id)))

            # nonlin_layers.append(SLogGate(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))
            # nonlin_layers.append(PReLU(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))
            nonlin_layers.append(PReLU(curr_c, curr_n, mode='spatial', name=str(layer_id)))

            interpolation_layers.append(AffineInterpolate(curr_c, curr_n, name=str(layer_id)))

        self.actnorm_layers = torch.nn.ModuleList(actnorm_layers)
        self.pre_interpolation_layers = torch.nn.ModuleList(pre_interpolation_layers)
        self.pre_nonlin_layers = torch.nn.ModuleList(pre_nonlin_layers)
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.nonlin_layers = torch.nn.ModuleList(nonlin_layers)
        self.interpolation_layers = torch.nn.ModuleList(interpolation_layers)
        self.squeeze_layer = Squeeze()

        self.c_out = curr_c
        self.n_out = curr_n
        print('\n**********************************************************\n')

    ################################################################################################

    def dequantize(self, x, quantization_levels=255.):
        # https://arxiv.org/pdf/1511.01844.pdf
        scale = 1/quantization_levels
        uniform_sample = self.uniform_dist.sample(x.shape)[..., 0]
        return x+scale*uniform_sample

    def jacobian(self, x):
        dummy_optimizer = torch.optim.Adam(self.parameters())
        x.requires_grad = True

        func_to_J = self.transform_with_logdet
        z, _ = func_to_J(x)
        assert (len(z.shape) == 4 and len(x.shape) == 4)
        assert (z.shape[0] == x.shape[0])
        assert (np.prod(z.shape[1:]) == np.prod(x.shape[1:]))

        J = np.zeros(z.shape+x.shape[1:])
        for i in range(z.shape[1]):
            for a in range(z.shape[2]):
                for b in range(z.shape[3]):
                    print(i, a, b)
                    dummy_optimizer.zero_grad() # zero the parameter gradients
                    if x.grad is not None: x.grad.zero_()

                    z, _ = func_to_J(x)
                    loss = torch.sum(z[:, i, a, b])
                    loss.backward()
                    J[:, i, a, b, ...] = helper.to_numpy(x.grad)

        J_flat = J.reshape(z.shape[0], np.prod(z.shape[1:]), np.prod(x.shape[1:]))
        return J, J_flat

    ################################################################################################

    def compute_uninitialized_actnorm_stats(self, data_loader, setup_mode='Training', n_batches=500, sub_image=None):
        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Mean computation.' )

        n_examples = 0
        accum_mean = None
        for i, curr_batch_size, batch_np in data_loader:     
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            actnorm_out, actnorm_object_mean = self.transform_with_logdet(image, initialization=True)
            if type(actnorm_object_mean) is not Actnorm: return None, None, None, None, None

            actnorm_out = helper.to_numpy(actnorm_out)
            if actnorm_object_mean.mode == 'spatial': curr_mean = actnorm_out.sum(0)
            elif actnorm_object_mean.mode == 'non-spatial': curr_mean = actnorm_out.mean(axis=(2, 3)).sum(0)

            if accum_mean is None: accum_mean = curr_mean
            else: accum_mean += curr_mean
            n_examples += actnorm_out.shape[0]

        mean = accum_mean/n_examples

        data_loader.setup(setup_mode, randomized=False, verbose=False)
        print('Std computation.' )
        
        n_examples = 0
        accum_var = None
        for i, curr_batch_size, batch_np in data_loader:  
            if n_batches is not None and i > n_batches: break
            image_np = batch_np['Image']
            if sub_image is not None: image_np = image_np[:, :sub_image[0], :sub_image[1], :sub_image[2]]
            image = helper.cuda(torch.from_numpy(image_np))

            actnorm_out, actnorm_object_var = self.transform_with_logdet(image, initialization=True)
            if type(actnorm_object_var) is not Actnorm: return None, None, None, None, None

            actnorm_out = helper.to_numpy(actnorm_out)
            if actnorm_object_var.mode == 'spatial': curr_var = ((actnorm_out-mean[np.newaxis, :, :, :])**2).sum(0)
            elif actnorm_object_var.mode == 'non-spatial':
                curr_var = ((actnorm_out-mean[np.newaxis, :, np.newaxis, np.newaxis])**2).mean(axis=(2, 3)).sum(0)

            if accum_var is None: accum_var = curr_var
            else: accum_var += curr_var
            n_examples += actnorm_out.shape[0]
        
        var = accum_var/n_examples
        std = np.sqrt(var)
        log_std = 0.5*np.log(var)
        bias = -mean/(np.exp(log_std)+1e-5)
        log_scale = -log_std

        assert (actnorm_object_mean == actnorm_object_var)
        if actnorm_object_var.mode == 'spatial': 
            bias = bias[np.newaxis, :, :, :].astype(np.float32)
            log_scale = log_scale[np.newaxis, :, :, :].astype(np.float32)
        elif actnorm_object_var.mode == 'non-spatial':
            bias = bias[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
            log_scale = log_scale[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)

        return actnorm_object_var, bias, log_scale, mean, std

    def set_actnorm_parameters(self, data_loader, setup_mode='Training', n_batches=500, test_normalization=True, sub_image=None):
        while True:
            print('\n')
            actnorm_object, actnorm_bias_np, actnorm_log_scale_np, _, _ = \
                self.compute_uninitialized_actnorm_stats(data_loader, setup_mode, n_batches, sub_image)
            if actnorm_object is None: break

            actnorm_object.set_parameters(actnorm_bias_np, actnorm_log_scale_np)
            print(actnorm_object.name + ' is initialized.\n')

            if test_normalization:
                print('Testing normalization: ')
                actnorm_object_test, actnorm_bias_np, actnorm_log_scale_np, _, _ = \
                    self.compute_uninitialized_actnorm_stats(data_loader, setup_mode, n_batches, sub_image)
                assert (np.abs(actnorm_bias_np).max() < 1e-4 and np.abs(actnorm_log_scale_np).max() < 1e-4)
                assert (actnorm_object_test == actnorm_object)
                print('PASSED: ' + actnorm_object_test.name + ' is normalizing.\n')

            actnorm_object.set_initialized()

    ################################################################################################

    def compute_normal_log_pdf(self, z):
        return (-0.5*np.log(2*np.pi)-0.5*(z*z)).sum(axis=[1, 2, 3])
        # return self.normal_dist.log_prob(z).sum(axis=[1, 2, 3])

    def sample_z(self, n_samples=10):
        with torch.no_grad():
            return self.normal_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]

    def sample_sharper_z(self, n_samples=10):
        with torch.no_grad():
            return self.normal_sharper_dist.sample([n_samples, self.c_out, self.n_out, self.n_out])[..., 0]

    def sample_x(self, n_samples=10):
        with torch.no_grad():
            return self.inverse_transform(self.sample_z(n_samples))

    def sample_sharper_x(self, n_samples=10):
        with torch.no_grad():
            return self.inverse_transform(self.sample_sharper_z(n_samples))

    ################################################################################################

    def transform_with_logdet(self, x, initialization=False):
        actnorm_logdets, pre_interpolation_logdets, pre_nonlin_logdets = [], [], []
        conv_logdets, interpolation_logdets, nonlin_logdets = [], [], []

        x = x-0.5
        curr_y = x
        for layer_id, k in enumerate(self.k_list): 
            for squeeze_i in range(self.squeeze_list[layer_id]): curr_y, _ = self.squeeze_layer.transform_with_logdet(curr_y)

            curr_y, actnorm_logdet = self.actnorm_layers[layer_id].transform_with_logdet(curr_y)
            actnorm_logdets.append(actnorm_logdet)

            if initialization and not self.actnorm_layers[layer_id].initialized:
                return curr_y, self.actnorm_layers[layer_id]

            curr_y, pre_interpolation_logdet = self.pre_interpolation_layers[layer_id].transform_with_logdet(curr_y)
            pre_interpolation_logdets.append(pre_interpolation_logdet)

            curr_y, pre_nonlin_logdet = self.pre_nonlin_layers[layer_id].transform_with_logdet(curr_y)
            pre_nonlin_logdets.append(pre_nonlin_logdet)

            curr_y, conv_logdet = self.conv_layers[layer_id].transform_with_logdet(curr_y)
            conv_logdets.append(conv_logdet)

            curr_y, nonlin_logdet = self.nonlin_layers[layer_id].transform_with_logdet(curr_y)
            nonlin_logdets.append(nonlin_logdet)
            
            curr_y, interpolation_logdet = self.interpolation_layers[layer_id].transform_with_logdet(curr_y)
            interpolation_logdets.append(interpolation_logdet)

        y = curr_y
        total_logdet = sum(actnorm_logdets)+sum(pre_interpolation_logdets)+sum(pre_nonlin_logdets)+\
                        sum(conv_logdets)+sum(nonlin_logdets)+sum(interpolation_logdets)
        return y, total_logdet

    def inverse_transform(self, y):
        with torch.no_grad():

            curr_y = y
            for layer_id in range(len(self.k_list)-1, -1,-1):
                curr_y = self.interpolation_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.nonlin_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.conv_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.pre_nonlin_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.pre_interpolation_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.actnorm_layers[layer_id].inverse_transform(curr_y)
                for squeeze_i in range(self.squeeze_list[layer_id]): curr_y = self.squeeze_layer.inverse_transform(curr_y)

            x = curr_y
            x = x+0.5   
            return x

    def forward(self, x, dequantize=True):
        if dequantize: x = self.dequantize(x)
        z, logdet = self.transform_with_logdet(x)
        log_pdf_z = self.compute_normal_log_pdf(z)
        log_pdf_x = log_pdf_z + logdet
        return z, x, logdet, log_pdf_z, log_pdf_x








