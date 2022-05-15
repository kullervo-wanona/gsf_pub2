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
import helper
import glob

main_dir = '/Volumes/GoogleDrive/My Drive/CollabResources/ExperimentalResults/samples_from_schur/'
layer_dirs = glob.glob(main_dir+'layers/layer_*')
image_ind_list = [3, 4, 5, 6, 7]

n_layers = len(layer_dirs)
n_samples = len(image_ind_list)

skip_layers = 2
col_list = []
for sample_id, image_ind in enumerate(image_ind_list):
    image_name = 'sample_real_'+str(image_ind)+'.png'
    
    real_image_path = main_dir + 'test_real/'+image_name
    real_image_np = helper.load_image(real_image_path, size=None)
    row_list = [real_image_np[np.newaxis]]
    for layer_id in range(n_layers):
        if layer_id % skip_layers != 0: continue
        curr_layer_dir = main_dir+'layers/layer'+'_'+ str(layer_id) +'_test/'
        image_path = curr_layer_dir+image_name
        print(image_path)
        image_np = helper.load_image(image_path, size=None)
        row_list.append(image_np[np.newaxis])
    row = np.concatenate(row_list, 0)
    col_list.append(row[np.newaxis])
raster = np.concatenate(col_list, 0)

helper.visualize_image_matrix(raster, block_size=None, max_rows=None, padding=[4, 4], save_path_list=['./blah.png'], verbosity_level=4)

trace()








    