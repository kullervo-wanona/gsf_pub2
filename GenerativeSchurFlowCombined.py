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
from Transforms import MultiChannel2DCircularConv, AffineBounded, PReLU, FixedSLogGate, Actnorm, ActnormNoLearning, Squeeze
from ConditionalTransforms import CondAffine, CondAffineBounded

class GenerativeSchurFlowPart1(torch.nn.Module):
    def __init__(self, c_in, n_in, k_list, squeeze_list, final_actnorm=False):
        super().__init__()
        assert (len(k_list) == len(squeeze_list))
        self.name = 'GenerativeSchurFlowPart1'
        self.n_in = n_in
        self.c_in = c_in
        self.k_list = k_list
        self.squeeze_list = squeeze_list
        self.final_actnorm = final_actnorm
        self.nonlin_class = PReLU
        # self.nonlin_class = FixedSLogGate
        self.n_layers = len(self.k_list)
        self.max_n_layer_conv = 5

        self.squeeze_layer = Squeeze()

        print('\n**********************************************************')
        print('Creating GenerativeSchurFlow: ')
        print('**********************************************************\n')
        actnorm_layers, conv_layers, conv_nonlin_layers, post_affine_layers, post_nonlin_layers = [], [], [], [], []

        accum_squeeze = 0
        for layer_id in range(self.n_layers):
            accum_squeeze += self.squeeze_list[layer_id]
            curr_c = self.c_in*(4**accum_squeeze)
            curr_n = self.n_in//(2**accum_squeeze)
            curr_k = self.k_list[layer_id]
            layer_conv_k_list = np.unique(np.round(np.arange(1, curr_k+(curr_k-1)/(self.max_n_layer_conv-1), (curr_k-1)/(self.max_n_layer_conv-1)))).astype(int).tolist()
            print('Layer '+str(layer_id)+': c='+str(curr_c)+', n='+str(curr_n)+', k list='+str(layer_conv_k_list))
            assert (curr_n >= curr_k)

            layer_actnorms, layer_convs, layer_conv_nonlins = [], [], []
            for layer_conv_id, layer_conv_k in enumerate(layer_conv_k_list):

                layer_actnorms.append(Actnorm(curr_c, curr_n, name='ConvFlow_actnorm_'+str(layer_id)+'_'+str(layer_conv_id)))
                layer_convs.append(MultiChannel2DCircularConv(
                    curr_c, curr_n, layer_conv_k, kernel_init='I + he_uniform', 
                    bias_mode='non-spatial', scale_mode='no-scale', name='ConvFlow_conv_'+str(layer_id)+'_'+str(layer_conv_id)))
                layer_conv_nonlins.append(self.nonlin_class(curr_c, curr_n, mode='non-spatial', name='ConvFlow_conv_nonlin_'+str(layer_id)+'_'+str(layer_conv_id)))

            actnorm_layers.append(layer_actnorms)
            conv_layers.append(layer_convs)
            conv_nonlin_layers.append(layer_conv_nonlins)
            
            post_affine_layers.append(AffineBounded(curr_c, curr_n, name='ConvFlow_post_affine_'+str(layer_id)))
            post_nonlin_layers.append(self.nonlin_class(curr_c, curr_n, name='ConvFlow_post_nonlin_'+str(layer_id)))

        self.actnorm_layers = actnorm_layers
        self.conv_layers = conv_layers
        self.conv_nonlin_layers = conv_nonlin_layers

        self.flat_actnorm_layers = torch.nn.ModuleList([e for layer_actnorms_ee in self.actnorm_layers for e in layer_actnorms_ee])
        self.flat_conv_layers = torch.nn.ModuleList([e for layer_convs_ee in self.conv_layers for e in layer_convs_ee])
        self.flat_conv_nonlin_layers = torch.nn.ModuleList([e for layer_conv_nonlins_ee in self.conv_nonlin_layers for e in layer_conv_nonlins_ee])

        self.post_affine_layers = torch.nn.ModuleList(post_affine_layers)
        self.post_nonlin_layers = torch.nn.ModuleList(post_nonlin_layers)

        self.c_out = curr_c
        self.n_out = curr_n
        print('\n**********************************************************\n')

    ################################################################################################

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

    def transform_with_logdet(self, x, initialization=False):
        actnorm_logdets, conv_logdets, conv_nonlin_logdets, = [], [], []
        post_nonlin_logdets, post_affine_logdets = [], []

        curr_y = x
        for layer_id, k in enumerate(self.k_list): 
            for squeeze_i in range(self.squeeze_list[layer_id]): curr_y, _ = self.squeeze_layer.transform_with_logdet(curr_y)

            for layer_conv_id in range(len(self.conv_layers[layer_id])):

                curr_y, actnorm_logdet = self.actnorm_layers[layer_id][layer_conv_id].transform_with_logdet(curr_y)
                actnorm_logdets.append(actnorm_logdet)

                if initialization and not self.actnorm_layers[layer_id][layer_conv_id].initialized:
                    return curr_y, self.actnorm_layers[layer_id][layer_conv_id]

                curr_y, conv_logdet = self.conv_layers[layer_id][layer_conv_id].transform_with_logdet(curr_y)
                conv_logdets.append(conv_logdet)

                curr_y, conv_nonlin_logdet = self.conv_nonlin_layers[layer_id][layer_conv_id].transform_with_logdet(curr_y)
                conv_nonlin_logdets.append(conv_nonlin_logdet)

            curr_y, post_affine_logdet = self.post_affine_layers[layer_id].transform_with_logdet(curr_y)
            post_affine_logdets.append(post_affine_logdet)

            curr_y, post_nonlin_logdet = self.post_nonlin_layers[layer_id].transform_with_logdet(curr_y)
            post_nonlin_logdets.append(post_nonlin_logdet)

        for i in range(sum(self.squeeze_list)): curr_y = self.squeeze_layer.inverse_transform(curr_y)
        y = curr_y

        total_logdet = sum(actnorm_logdets)+sum(conv_logdets)+sum(conv_nonlin_logdets)+\
                       sum(post_affine_logdets)+sum(post_nonlin_logdets)
        return y, total_logdet

    def inverse_transform(self, y):
        with torch.no_grad():

            curr_y = y
            for i in range(sum(self.squeeze_list)): curr_y, _ = self.squeeze_layer.transform_with_logdet(curr_y)

            for layer_id in range(len(self.k_list)-1, -1,-1):

                curr_y = self.post_nonlin_layers[layer_id].inverse_transform(curr_y)
                curr_y = self.post_affine_layers[layer_id].inverse_transform(curr_y)

                for layer_conv_id in range(len(self.conv_layers[layer_id])-1, -1,-1):
                    layer_conv_nonlin = self.conv_nonlin_layers[layer_id][layer_conv_id]
                    layer_conv = self.conv_layers[layer_id][layer_conv_id]
                    layer_actnorm = self.actnorm_layers[layer_id][layer_conv_id]
                    
                    curr_y = layer_conv_nonlin.inverse_transform(curr_y)
                    curr_y = layer_conv.inverse_transform(curr_y)
                    curr_y = layer_actnorm.inverse_transform(curr_y)

                for squeeze_i in range(self.squeeze_list[layer_id]): curr_y = self.squeeze_layer.inverse_transform(curr_y)

            x = curr_y
            return x

    ################################################################################################

class ConditionalSchurTransformPart2(torch.nn.Module):
    def __init__(self, c_in, n_in, n_layers):
        super().__init__()
        self.name = 'ConditionalSchurTransformPart2'
        self.n_in = n_in
        self.c_in = c_in
        self.n_layers = n_layers

        print('\n**********************************************************')
        print('Creating ConditionalSchurTransform: ')
        print('**********************************************************\n')

        actnorm_layers, affine_layers, nonlin_layers = [], [], []

        self.non_spatial_conditional_transforms = {}
        self.spatial_conditional_transforms = {}

        curr_c = self.c_in
        curr_n = self.n_in
        for layer_id in range(self.n_layers):
            print('Layer '+str(layer_id)+': c='+str(curr_c)+', n='+str(curr_n))

            actnorm_layers.append(Actnorm(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))

            affine_layer = CondAffine(curr_c, curr_n, bias_mode='spatial', scale_mode='spatial', name=str(layer_id))
            # affine_layer = CondAffineBounded(curr_c, curr_n, bias_mode='spatial', scale_mode='spatial', name=str(layer_id))
            self.non_spatial_conditional_transforms[affine_layer.name] = affine_layer
            affine_layers.append(affine_layer)

            # nonlin_layers.append(PReLU(curr_c, curr_n, mode='non-spatial', name=str(layer_id)))

        self.actnorm_layers = torch.nn.ModuleList(actnorm_layers)
        self.affine_layers = torch.nn.ModuleList(affine_layers)
        # self.nonlin_layers = torch.nn.ModuleList(nonlin_layers)

        self.c_out = curr_c
        self.n_out = curr_n
        self.non_spatial_n_cond_params, self.non_spatial_cond_param_sizes_list = self.non_spatial_conditional_param_sizes()
        self.spatial_cond_param_shape, self.spatial_cond_param_sizes_list = self.spatial_conditional_param_sizes()

        print('\n**********************************************************\n')
        
    ################################################################################################

    def jacobian(self, x):
        dummy_optimizer = torch.optim.Adam(self.parameters())
        x.requires_grad = True

        func_to_J = self.transform
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

    ###############################################################################################

    def non_spatial_conditional_param_sizes(self):
        param_sizes_list = []
        total_n_params = 0
        for transform_name in sorted(self.non_spatial_conditional_transforms):
            for param_name in sorted(self.non_spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_sizes_list.append((transform_name + '__' + param_name, None))
                else:
                    curr_n_param = np.prod(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:])
                    param_sizes_list.append((transform_name + '__' + param_name, 
                        self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name]))
                    total_n_params += curr_n_param
        return total_n_params, param_sizes_list

    def non_spatial_conditional_param_assignments(self, tensor):
        param_assignments = {}
        total_n_params = 0
        for transform_name in sorted(self.non_spatial_conditional_transforms):
            param_assignments[transform_name] = {}
            for param_name in sorted(self.non_spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_assignments[transform_name][param_name] = None
                else:
                    curr_n_param = np.prod(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:])
                    curr_param_flat = tensor[:, total_n_params:total_n_params+curr_n_param]
                    curr_param = curr_param_flat.reshape(self.non_spatial_conditional_transforms[transform_name].parameter_sizes[param_name])
                    param_assignments[transform_name][param_name] = curr_param
                    total_n_params += curr_n_param
        return total_n_params, param_assignments

    def spatial_conditional_param_sizes(self):
        param_sizes_list = []
        total_param_shape = None
        for transform_name in sorted(self.spatial_conditional_transforms):
            for param_name in sorted(self.spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_sizes_list.append((transform_name + '__' + param_name, None))
                else:
                    curr_param_shape = self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:]
                    param_sizes_list.append((transform_name + '__' + param_name, 
                        self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name]))
                    if total_param_shape is None: total_param_shape = curr_param_shape
                    else:
                        try:
                            assert (total_param_shape[1:] == curr_param_shape[1:])
                        except:
                            trace()
                        total_param_shape[0] += curr_param_shape[0]
        if total_param_shape is None: total_param_shape = [0, 0, 0]
        return total_param_shape, param_sizes_list

    def spatial_conditional_param_assignments(self, tensor):
        param_assignments = {}
        total_param_shape = None
        for transform_name in sorted(self.spatial_conditional_transforms):
            param_assignments[transform_name] = {}
            for param_name in sorted(self.spatial_conditional_transforms[transform_name].parameter_sizes):
                if self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name] is None:
                    param_assignments[transform_name][param_name] = None
                else:
                    curr_param_shape = self.spatial_conditional_transforms[transform_name].parameter_sizes[param_name][1:]
                    if total_param_shape is None:
                        curr_param = tensor[:, :curr_param_shape[0], :, :]
                    else:
                        curr_param = tensor[:, total_param_shape[0]:total_param_shape[0]+curr_param_shape[0], :, :]
                    param_assignments[transform_name][param_name] = curr_param

                    if total_param_shape is None: total_param_shape = curr_param_shape
                    else:
                        assert (total_param_shape[1:] == curr_param_shape[1:])
                        total_param_shape[0] += curr_param_shape[0]        
        return total_param_shape, param_assignments

    def transform_with_logdet(self, x, non_spatial_param, spatial_param, initialization=False):
        if non_spatial_param is not None:
            _, non_spatial_param_assignments = self.non_spatial_conditional_param_assignments(non_spatial_param)
        if spatial_param is not None:
            _, spatial_param_assignments = self.spatial_conditional_param_assignments(spatial_param)

        actnorm_logdets, affine_logdets, nonlin_logdets = [], [], []

        curr_y = x
        for layer_id in range(self.n_layers):

            curr_y, actnorm_logdet = self.actnorm_layers[layer_id].transform_with_logdet(curr_y)
            actnorm_logdets.append(actnorm_logdet)

            if initialization and not self.actnorm_layers[layer_id].initialized: 
                return curr_y, self.actnorm_layers[layer_id]

            curr_params = non_spatial_param_assignments[self.affine_layers[layer_id].name]
            affine_bias, affine_pre_scale = curr_params["bias"], curr_params["pre_scale"]
            curr_y, affine_logdet = self.affine_layers[layer_id].transform_with_logdet(curr_y, affine_bias, affine_pre_scale)
            affine_logdets.append(affine_logdet)

            # curr_y, nonlin_logdet = self.nonlin_layers[layer_id].transform_with_logdet(curr_y)
            # nonlin_logdets.append(nonlin_logdet)

        y = curr_y
        total_log_det = sum(actnorm_logdets)+sum(affine_logdets)+sum(nonlin_logdets)
        return y, total_log_det

    def inverse_transform(self, y, non_spatial_param, spatial_param):
        with torch.no_grad():
            if non_spatial_param is not None:
                _, non_spatial_param_assignments = self.non_spatial_conditional_param_assignments(non_spatial_param)
            if spatial_param is not None:
                _, spatial_param_assignments = self.spatial_conditional_param_assignments(spatial_param)

            curr_y = y
            for layer_id in range(self.n_layers-1, -1,-1):

                # curr_y = self.nonlin_layers[layer_id].inverse_transform(curr_y)

                curr_params = non_spatial_param_assignments[self.affine_layers[layer_id].name]
                affine_bias, affine_pre_scale = curr_params["bias"], curr_params["pre_scale"]
                curr_y = self.affine_layers[layer_id].inverse_transform(curr_y, affine_bias, affine_pre_scale)
                
                curr_y = self.actnorm_layers[layer_id].inverse_transform(curr_y)
                
            x = curr_y
            return x
    

    ################################################################################################

class ViewLayer(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
        
class GenerativeConditionalSchurFlow(torch.nn.Module):
    def __init__(self, c_in, n_in, n_cond_blocks):
        super().__init__()

        self.name = 'GenerativeConditionalSchurFlow'
        self.c_in = c_in
        self.n_in = n_in

        self.n_cond_blocks = n_cond_blocks

        self.squeeze_layers = [Squeeze(chan_mode='input_channels_apart', spatial_mode='tl-br-tr-bl'), 
                               Squeeze(chan_mode='input_channels_adjacent', spatial_mode='tl-br-tr-bl'), 
                               Squeeze(chan_mode='input_channels_apart', spatial_mode='tl-tr-bl-br'),
                               Squeeze(chan_mode='input_channels_adjacent', spatial_mode='tl-tr-bl-br'),]   

        update_cond_schur_transform_list = [ConditionalSchurTransformPart2(c_in=self.c_in*4//2, n_in=self.n_in//2, n_layers=1) for block_id in range(self.n_cond_blocks)]
        self.update_cond_schur_transform_list = torch.nn.ModuleList(update_cond_schur_transform_list)

        base_cond_schur_transform_list = [ConditionalSchurTransformPart2(c_in=self.c_in*4//2, n_in=self.n_in//2, n_layers=1) for block_id in range(self.n_cond_blocks)]
        self.base_cond_schur_transform_list = torch.nn.ModuleList(base_cond_schur_transform_list)

        base_spatial_cond_nets, base_non_spatial_cond_nets = [], []
        update_spatial_cond_nets, update_non_spatial_cond_nets = [], []

        self.FC_main_cond_net_c_out = 512
        self.main_cond_net = self.create_fc_main_cond_net(c_in=(self.c_in*4//2), n_in=self.n_in//2, c_out=self.FC_main_cond_net_c_out)
        for block_id in range(self.n_cond_blocks):
            print('Base net, block '+ str(block_id) +' spatial_cond_param_shape[0]: ' + str(self.update_cond_schur_transform_list[block_id].spatial_cond_param_shape[0]))
            print('Base net, block '+ str(block_id) +' non_spatial_n_cond_params: ' + str(self.update_cond_schur_transform_list[block_id].non_spatial_n_cond_params))

            base_spatial_cond_nets.append(self.create_fc_spatial_cond_net(c_in=self.FC_main_cond_net_c_out, n_out=self.n_in//2, c_out=self.update_cond_schur_transform_list[block_id].spatial_cond_param_shape[0]))
            base_non_spatial_cond_nets.append(self.create_fc_non_spatial_cond_net(c_in=self.FC_main_cond_net_c_out, c_out=self.update_cond_schur_transform_list[block_id].non_spatial_n_cond_params))

            print('Update net, block '+ str(block_id) +' spatial_cond_param_shape[0]: ' + str(self.base_cond_schur_transform_list[block_id].spatial_cond_param_shape[0]))
            print('Update net, block '+ str(block_id) +' non_spatial_n_cond_params: ' + str(self.base_cond_schur_transform_list[block_id].non_spatial_n_cond_params))

            update_spatial_cond_nets.append(self.create_fc_spatial_cond_net(c_in=self.FC_main_cond_net_c_out, n_out=self.n_in//2, c_out=self.base_cond_schur_transform_list[block_id].spatial_cond_param_shape[0]))
            update_non_spatial_cond_nets.append(self.create_fc_non_spatial_cond_net(c_in=self.FC_main_cond_net_c_out, c_out=self.base_cond_schur_transform_list[block_id].non_spatial_n_cond_params))
    
        self.base_spatial_cond_nets = torch.nn.ModuleList(base_spatial_cond_nets)
        self.base_non_spatial_cond_nets = torch.nn.ModuleList(base_non_spatial_cond_nets)

        self.update_spatial_cond_nets = torch.nn.ModuleList(update_spatial_cond_nets)
        self.update_non_spatial_cond_nets = torch.nn.ModuleList(update_non_spatial_cond_nets)

        self.c_out = self.c_in
        self.n_out = self.n_in

    ################################################################################################

    def create_fc_main_cond_net(self, c_in, n_in, c_out, channel_multiplier=8):
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(c_in*n_in*n_in, c_out),
            torch.nn.LayerNorm(c_out),
            torch.nn.ReLU(True),
            torch.nn.Linear(c_out, c_out),
            torch.nn.LayerNorm(c_out),
            torch.nn.ReLU(True),
            )
        net = helper.cuda(net)
        # out = net(torch.rand((10, c_in, n_in, n_in)))
        # trace()
        return net

    def create_fc_spatial_cond_net(self, c_in, n_out, c_out, channel_multiplier=8):
        if c_out == 0: return None
        net = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_out*n_out*n_out),
            ViewLayer(shape=[-1, c_out, n_out, n_out])
            )
        net = helper.cuda(net)
        # out = net(torch.rand((10, 512)))
        # trace()
        return net

    def create_fc_non_spatial_cond_net(self, c_in, c_out, channel_multiplier=8):
        if c_out == 0: return None
        net = torch.nn.Sequential(
            torch.nn.Linear(c_in, c_out),
            )

        net = helper.cuda(net)
        # out = net(torch.rand((10, 512)))
        # trace()
        return net

    ################################################################################################

    def base_cond_net_forward(self, x, block_id):
        main_cond = self.main_cond_net(x)
        non_spatial_param = None
        if self.base_non_spatial_cond_nets[block_id] is not None: 
            non_spatial_param = self.base_non_spatial_cond_nets[block_id](main_cond)
        spatial_param = None
        if self.base_spatial_cond_nets[block_id] is not None: 
            spatial_param = self.base_spatial_cond_nets[block_id](main_cond)
        return non_spatial_param, spatial_param

    def update_cond_net_forward(self, x, block_id):
        main_cond = self.main_cond_net(x)
        non_spatial_param = None
        if self.update_non_spatial_cond_nets[block_id] is not None: 
            non_spatial_param = self.update_non_spatial_cond_nets[block_id](main_cond)
        spatial_param = None
        if self.update_spatial_cond_nets[block_id] is not None: 
            spatial_param = self.update_spatial_cond_nets[block_id](main_cond)
        return non_spatial_param, spatial_param


################################################################################################

    def transform_with_logdet(self, x, initialization=False):
        all_logdets = []
        layer_input = x

        for block_id in range(self.n_cond_blocks):
            # print('forward:', block_id)

            layer_input_squeezed, _ = self.squeeze_layers[block_id % len(self.squeeze_layers)].transform_with_logdet(layer_input)
            curr_base, curr_update = layer_input_squeezed[:, :layer_input_squeezed.shape[1]//2], layer_input_squeezed[:, layer_input_squeezed.shape[1]//2:]

            non_spatial_param, spatial_param = self.base_cond_net_forward(curr_base, block_id)
            new_update, update_logdet = self.update_cond_schur_transform_list[block_id].transform_with_logdet(curr_update, non_spatial_param, spatial_param, initialization)
            if type(update_logdet) is Actnorm or type(update_logdet) is ActnormNoLearning: return new_update, update_logdet # init run unparameterized actnorm

            non_spatial_param, spatial_param = self.update_cond_net_forward(new_update, block_id)            
            new_base, base_logdet = self.base_cond_schur_transform_list[block_id].transform_with_logdet(curr_base, non_spatial_param, spatial_param, initialization)
            if type(base_logdet) is Actnorm or type(base_logdet) is ActnormNoLearning: return new_base, base_logdet # init run unparameterized actnorm

            curr_lodget = update_logdet+base_logdet
            all_logdets.append(curr_lodget)

            layer_out_squeezed = torch.concat([new_base, new_update], axis=1)
            layer_out = self.squeeze_layers[block_id % len(self.squeeze_layers)].inverse_transform(layer_out_squeezed)

            layer_input = layer_out

        z = layer_out
        total_lodget = sum(all_logdets)
        return z, total_lodget

    def inverse_transform(self, z):
        with torch.no_grad():

            layer_out = z
            for block_id in range(self.n_cond_blocks-1, -1, -1):
                # print('inverting:', block_id)
                layer_out_squeezed, _ = self.squeeze_layers[block_id % len(self.squeeze_layers)].transform_with_logdet(layer_out)
                curr_base, curr_update = layer_out_squeezed[:, :layer_out_squeezed.shape[1]//2], layer_out_squeezed[:, layer_out_squeezed.shape[1]//2:]

                non_spatial_param, spatial_param = self.update_cond_net_forward(curr_update, block_id)
                old_base = self.base_cond_schur_transform_list[block_id].inverse_transform(curr_base, non_spatial_param, spatial_param)

                non_spatial_param, spatial_param = self.base_cond_net_forward(old_base, block_id)
                old_update = self.update_cond_schur_transform_list[block_id].inverse_transform(curr_update, non_spatial_param, spatial_param)
                
                layer_input_squeezed = torch.concat([old_base, old_update], axis=1)
                layer_input = self.squeeze_layers[block_id % len(self.squeeze_layers)].inverse_transform(layer_input_squeezed)
                layer_out = layer_input

            x = layer_input     
            return x 


class GenerativeSchurFlowCombined(torch.nn.Module):
    def __init__(self, c_in, n_in, conv_k_list, conv_squeeze_list, n_cond_blocks):
        super().__init__()

        self.name = 'GenerativeSchurFlowCombined'
        self.c_in = c_in
        self.n_in = n_in

        self.conv_k_list = conv_k_list
        self.conv_squeeze_list = conv_squeeze_list
        self.n_cond_blocks = n_cond_blocks

        self.uniform_dist = torch.distributions.Uniform(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([1.0])))
        self.normal_sharper_dist = torch.distributions.Normal(helper.cuda(torch.tensor([0.0])), helper.cuda(torch.tensor([0.7])))

        # self.conv_flow_net = GenerativeSchurFlowPart1(c_in, n_in, k_list=self.conv_k_list, squeeze_list=self.conv_squeeze_list)
        self.conditional_flow_net = GenerativeConditionalSchurFlow(c_in, n_in, n_cond_blocks=self.n_cond_blocks)

        self.c_out = self.c_in
        self.n_out = self.n_in

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
            if type(actnorm_object_mean) is not Actnorm and type(actnorm_object_mean) is not ActnormNoLearning: return None, None, None, None, None

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
            if type(actnorm_object_var) is not Actnorm and type(actnorm_object_var) is not ActnormNoLearning: return None, None, None, None, None

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
                actnorm_object_test, actnorm_bias_np_test, actnorm_log_scale_np_test, _, _ = \
                    self.compute_uninitialized_actnorm_stats(data_loader, setup_mode, n_batches, sub_image)
                
                assert (np.abs(actnorm_bias_np_test).max() < 1e-4 and np.abs(actnorm_log_scale_np_test).max() < 1e-4)
                assert (actnorm_object_test == actnorm_object)
                print('PASSED: ' + actnorm_object_test.name + ' is normalizing.\n')

            actnorm_object.set_initialized()

    ###############################################################################################
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
    ###############################################################################################

    def transform_with_logdet(self, x, initialization=False):
        all_logdets = []

        x = x-0.5

        # z_first, logdet_first = self.conv_flow_net.transform_with_logdet(x, initialization)
        # if type(logdet_first) is Actnorm or type(logdet_first) is ActnormNoLearning: return z_first, logdet_first

        z_first, logdet_first = x, 0

        z_second, logdet_second = self.conditional_flow_net.transform_with_logdet(z_first, initialization)
        if type(logdet_second) is Actnorm or type(logdet_second) is ActnormNoLearning: return z_second, logdet_second

        z = z_second
        total_lodget = logdet_first + logdet_second
        return z, total_lodget

    def inverse_transform(self, z):
        with torch.no_grad():
            z_second = z
            z_first = self.conditional_flow_net.inverse_transform(z_second)
            # x = self.conv_flow_net.inverse_transform(z_first)
            x = z_first

            x = x+0.5
            return x

    def forward(self, x, dequantize=True):
        if dequantize: x = self.dequantize(x)
        z, logdet = self.transform_with_logdet(x)
        log_pdf_z = self.compute_normal_log_pdf(z)
        log_pdf_x = log_pdf_z + logdet
        return z, x, logdet, log_pdf_z, log_pdf_x













    # def forward(self, x_input, dequantize=True):
    #     x = x_input
    #     if dequantize: x = self.dequantize(x)
    #     x = x-0.5
    #     z, logdet = self.transform_with_logdet(x)
    #     log_pdf_z = self.compute_normal_log_pdf(z)
    #     log_pdf_x = log_pdf_z + logdet
    #     for i in range(sum(self.squeeze_list)):
    #         z = self.squeeze_layer.inverse_transform(z)
    #     z = z+0.5
    #     return z, x_input, logdet, log_pdf_z, log_pdf_x




    # def transform_all_layers(self, x):
    #     with torch.no_grad():
    #         all_z = [x]
    #         x = x - 0.5
    #         layer_input = x
    #         for block_id in range(self.n_cond_blocks):

    #             layer_input_squeezed, _ = self.squeeze_layers[block_id % len(self.squeeze_layers)].transform_with_logdet(layer_input)
    #             curr_base, curr_update = layer_input_squeezed[:, :layer_input_squeezed.shape[1]//2], layer_input_squeezed[:, layer_input_squeezed.shape[1]//2:]

    #             non_spatial_param, spatial_param = self.base_cond_net_forward(curr_base, block_id)
    #             new_update, update_logdet = self.update_cond_schur_transform_list[block_id].transform_with_logdet(curr_update, non_spatial_param, spatial_param)

    #             non_spatial_param, spatial_param = self.update_cond_net_forward(new_update, block_id)            
    #             new_base, base_logdet = self.base_cond_schur_transform_list[block_id].transform_with_logdet(curr_base, non_spatial_param, spatial_param)

    #             layer_out_squeezed = torch.concat([new_base, new_update], axis=1)
    #             layer_out = self.squeeze_layers[block_id % len(self.squeeze_layers)].inverse_transform(layer_out_squeezed)

    #             layer_input = layer_out
    #             all_z.append(layer_out)

    #         return all_z

    # def inverse_transform_all_layers(self, z):
    #     with torch.no_grad():

    #         layer_out = z
    #         all_z = [z]
    #         for block_id in range(self.n_cond_blocks-1, -1, -1):
    #             layer_out_squeezed, _ = self.squeeze_layers[block_id % len(self.squeeze_layers)].transform_with_logdet(layer_out)
    #             curr_base, curr_update = layer_out_squeezed[:, :layer_out_squeezed.shape[1]//2], layer_out_squeezed[:, layer_out_squeezed.shape[1]//2:]

    #             non_spatial_param, spatial_param = self.update_cond_net_forward(curr_update, block_id)
    #             old_base = self.base_cond_schur_transform_list[block_id].inverse_transform(curr_base, non_spatial_param, spatial_param)

    #             non_spatial_param, spatial_param = self.base_cond_net_forward(old_base, block_id)
    #             old_update = self.update_cond_schur_transform_list[block_id].inverse_transform(curr_update, non_spatial_param, spatial_param)
                
    #             layer_input_squeezed = torch.concat([old_base, old_update], axis=1)
    #             layer_input = self.squeeze_layers[block_id % len(self.squeeze_layers)].inverse_transform(layer_input_squeezed)
    #             layer_out = layer_input
    #             all_z.append(layer_out)

    #         x = layer_input
    #         x = x + 0.5
    #         all_z[-1] = x
    #         return all_z


