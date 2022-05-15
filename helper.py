from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import Tracer
    trace = Tracer() #this one triggers the debugger
else:
    import ipdb
    trace = ipdb.set_trace

import os
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

# CUDA_FLAG = False
CUDA_FLAG = True

def cuda(x):
    global CUDA_FLAG
    if torch.cuda.is_available() and CUDA_FLAG: return x.to(device='cuda')
    # if torch.cuda.is_available() and CUDA_FLAG and not x.is_cuda: return x.to(device='cuda')
    else: return x

def cpu(x):
    global CUDA_FLAG
    # if torch.cuda.is_available() and CUDA_FLAG and x.is_cuda: return x.cpu()
    if torch.cuda.is_available() and CUDA_FLAG: return x.cpu()
    else: return x

def to_numpy(x):
    return cpu(x.detach()).numpy()

def display(mat, rescale=False):
    if rescale: mat = (mat-mat.min())/(mat.max()-mat.min())
    plt.imshow(np.clip(mat, 0, 1), interpolation='nearest')
    plt.draw()
    plt.pause(0.001)

def save_image(mat, path, rescale=False, resize=None):
    if rescale: mat = (mat-mat.min())/(mat.max()-mat.min())
    im = Image.fromarray((np.clip(mat, 0, 1)*255.).astype(np.uint8))
    if resize is not None: im = im.resize(resize, Image.NEAREST)
    im.save(path)

def load_image(path, size=None):
    im_Image_format = Image.open(path)
    if size is not None: im_Image_format = im_Image_format.resize(size, Image.BICUBIC)
    im = np.asarray(im_Image_format, dtype="uint8").astype(float)[:, :, :3]/255.0
    return im 

def vis_samples_np(samples, sample_dir, prefix=None, resize=None):
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    for i in range(samples.shape[0]):
        path = sample_dir + 'sample_' 
        if prefix is not None: path += prefix + '_'
        path += str(i) + '.png' 
        if samples[i].shape[0] == 1: curr_sample = np.transpose(np.concatenate([samples[i], samples[i], samples[i]], axis=0), [1, 2, 0])
        else: curr_sample = np.transpose(samples[i], [1, 2, 0])
        save_image(curr_sample, path, rescale=False, resize=resize)
        
def vectorize(tensor):
    # vectorize last d-1 dims, column indeces together (entire row), then row indices (entire channel), then channel indices (entire image)
    # return tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2]*tensor.shape[3])
    return tensor.reshape(tensor.shape[0], np.prod(tensor.shape[1:]))

def unvectorize(tensor, shape_list):
    # vectorize last d-1 dims, column indeces together (entire row), then row indices (entire channel), then channel indices (entire image)
    # return tensor.reshape(tensor.shape[0], n_in_chan, n_rows, n_cols)
    return tensor.reshape(tensor.shape[0], *shape_list)

def get_conv_initial_weight_kernel_np(kernel_shape, n_in_channels, n_out_channels, initializer_mode='he_uniform'):
    # Written with convolution in mind, using terminology related to convolutions.
    fan_in, fan_out = n_in_channels*np.prod(kernel_shape), n_out_channels*np.prod(kernel_shape)
    if 'uniform' in initializer_mode:
        if initializer_mode == 'uniform':
            uniform_scale = 0.05
        elif initializer_mode == 'lecun_uniform': # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
            uniform_scale = np.sqrt(3./fan_in)
        elif initializer_mode == 'xavier_uniform': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            uniform_scale = np.sqrt(6./(fan_in + fan_out))
        elif initializer_mode == 'he_uniform': # http://arxiv.org/abs/1502.01852
            uniform_scale = np.sqrt(6./fan_in)
        elif initializer_mode == 'he_uniform_2': # http://arxiv.org/abs/1502.01852
            uniform_scale = np.sqrt(1./fan_in)
        initial_weight_kernel = np.random.uniform(low=-uniform_scale, high=uniform_scale, 
            size=[n_out_channels, n_in_channels]+kernel_shape).astype(np.float32)

    elif 'gaussian' in initializer_mode:
        if initializer_mode == 'gaussian' or initializer_mode == 'truncated_gaussian':
            # gaussian_std = 0.02
            gaussian_std = 0.4
        elif initializer_mode == 'xavier_gaussian': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            gaussian_std = np.sqrt(2./(fan_in+fan_out))
        elif initializer_mode == 'he_gaussian': # http://arxiv.org/abs/1502.01852
            gaussian_std = np.sqrt(2./fan_in)
        initial_weight_kernel = np.random.randn(
            *([n_out_channels, n_in_channels]+kernel_shape)).astype(np.float32)*gaussian_std

    return initial_weight_kernel



def save_or_show_images(image_mat, path=None):
    assert (len(image_mat.shape) in [2, 3])

    if image_mat.dtype != np.uint8:
        assert (image_mat.min() >= 0 and image_mat.max() <= 1)
        image_mat = (image_mat*255.).astype('uint8')

    if len(image_mat.shape) == 2:
        image_object = Image.fromarray(image_mat, mode='L')
    else:
        assert (image_mat.shape[2] in [1, 3, 4])
        if image_mat.shape[2] == 1:
            image_object = Image.fromarray(image_mat[:, :, 0], mode='L')
        elif image_mat.shape[2] == 3:
            image_object = Image.fromarray(image_mat, mode='RGB')
        elif image_mat.shape[2] == 4:
            image_object = Image.fromarray(image_mat, mode='RGBA')
        
    if path is None: image_object.show()
    else: image_object.save(path)
    
def visualize_image_matrix(tensor_np, block_size=None, max_rows=None, padding=[4, 4], save_path_list=None, verbosity_level=4):
    assert (len(tensor_np.shape) == 4 or len(tensor_np.shape) == 5)
    assert (tensor_np.shape[-1] == 1 or tensor_np.shape[-1] == 3)
    image_size = list(tensor_np.shape[-3:])

    if block_size is not None:
        if len(tensor_np.shape) == 4:
            assert(tensor_np.shape[0] == np.prod(block_size))
            tensor_np = tensor_np.reshape(block_size+image_size)
        elif len(tensor_np.shape) == 5:
            assert (np.prod(tensor_np.shape[:2]) == np.prod(block_size))
            if tensor_np.shape[:2] != block_size:
                tensor_np = tensor_np.reshape(block_size+image_size)
    else:
        if len(tensor_np.shape) == 4:
            batch_size_sqrt_floor = int(np.floor(np.sqrt(tensor_np.shape[0])))
            block_size = [batch_size_sqrt_floor, batch_size_sqrt_floor]
            tensor_np = tensor_np[:np.prod(block_size), ...].reshape(block_size+image_size)
        elif len(tensor_np.shape) == 5:
            block_size = tensor_np.shape[:2]

    if max_rows is None: max_rows = tensor_np.shape[0]    
    canvas = np.ones([image_size[0]*min(block_size[0], max_rows)+ padding[0]*(min(block_size[0], max_rows)+1), 
                      image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])
    for i in range(min(block_size[0], max_rows)):
        start_coor = padding[0] + i*(image_size[0]+padding[0])
        for t in range(block_size[1]):
            y_start = (t+1)*padding[1]+t*image_size[1]
            canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  tensor_np[i][t]
    if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)

    if np.isnan(canvas).any(): 
        print('\n\n\nWarning: The image matrix contains values that are NaNs.\n\n\n')

    if (canvas.min() < 0) or (canvas.max() > 1):
        print('\n\n\nWarning: The image matrix contains values outside of the range [0, 1].')
        print('Canvas ranges: ['+str(canvas.min())+', '+str(canvas.max())+']')
        print('Clipping the image matrix values to [0, 1].\n\n\n')
        canvas = np.clip(canvas, 0., 1.)

    full_save_path_list = []
    if save_path_list is None:
        save_or_show_images(canvas)
    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            full_path = path+'_ImageMatrix.png'
            save_or_show_images(canvas, full_path)
            full_save_path_list.append(full_path)

    return full_save_path_list






    