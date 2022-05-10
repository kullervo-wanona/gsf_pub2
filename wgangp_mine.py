import os, sys
sys.path.append(os.getcwd())

from sys import platform
if 'linux' in platform: 
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

import time
# import tflib as lib
# import tflib.cifar10
# import tflib.plot
# import tflib.inception_score

import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import helper
from pathlib import Path

# # Download CIFAR-10 (Python version) at
# # https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# # extracted files here!
# DATA_DIR = 'cifar-10-batches-py/'
# if len(DATA_DIR) == 0:
#     raise Exception('Please specify path to data directory in gan_cifar.py!')

from GenerativeSchurFlow import GenerativeSchurFlow

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

from DataLoaders.CelebA.CelebA32Loader import DataLoader

train_data_loader = DataLoader(batch_size=BATCH_SIZE)
train_data_loader.setup('Training', randomized=True, verbose=True)
_, _, example_batch = next(train_data_loader) 

test_data_loader = DataLoader(batch_size=BATCH_SIZE)
test_data_loader.setup('Test', randomized=False, verbose=False)
_, _, example_test_batch = next(test_data_loader) 




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # preprocess_it = nn.Sequential(
        #     nn.Linear(128, 4 * 4 * 4 * DIM),
        #     # nn.BatchNorm2d(4 * 4 * 4 * DIM),
        #     nn.ReLU(True),
        # )

        # block1 = nn.Sequential(
        #     nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
        #     nn.BatchNorm2d(2 * DIM),
        #     nn.ReLU(True),
        # )
        # block2 = nn.Sequential(
        #     nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
        #     nn.BatchNorm2d(DIM),
        #     nn.ReLU(True),
        # )
        # deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        # self.preprocess_it = preprocess_it
        # self.block1 = block1
        # self.block2 = block2
        # self.deconv_out = deconv_out
        self.c_in = 3
        self.n_in = 32
        self.flow_net = GenerativeSchurFlow(self.c_in, self.n_in, k_list=[10]*7, squeeze_list=[0]*7)
        self.tanh = nn.Tanh()

    def forward(self, input):
        #batch x DIM input
        output, _ = self.flow_net.transform(input)
        # output = self.preprocess_it(input)
        # output = output.view(-1, 4 * DIM, 4, 4)
        # output = self.block1(output)
        # output = self.block2(output)
        # output = self.deconv_out(output)
        output = self.tanh(output)
        #batch x 3, 32, 32
        # return output.view(-1, 3, 32, 32)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

netG = Generator()
netD = Discriminator()
print(netG)
print(netD)

n_param = 0
for name, e in netG.named_parameters():
    print(name, e.requires_grad, e.shape)
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

n_param = 0
for e in netG.parameters():
    n_param += np.prod(e.shape)
print('Total number of parameters: ' + str(n_param))

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE)
    alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis] 
    # alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    # interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    with torch.no_grad():
        fixed_noise_128 = torch.randn(128, 3, 32, 32)
        if use_cuda:
            fixed_noise_128 = fixed_noise_128.cuda(gpu)
        # noisev = autograd.Variable(fixed_noise_128, volatile=True)
        noisev = fixed_noise_128

        samples = netG(noisev)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.cpu().data.numpy()
        
        helper.vis_samples_np(samples, sample_dir=str(Path.home())+'/ExperimentalResults/samples_from_wgan/sample/', prefix='sample', resize=[256, 256])

    # lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

# # For calculating inception score
# def get_inception_score(G, ):
#     all_samples = []
#     for i in xrange(10):
#         samples_100 = torch.randn(100, 128)
#         if use_cuda:
#             samples_100 = samples_100.cuda(gpu)
#         samples_100 = autograd.Variable(samples_100, volatile=True)
#         all_samples.append(G(samples_100).cpu().data.numpy())

#     all_samples = np.concatenate(all_samples, axis=0)
#     all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
#     all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
#     return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
# train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)
# def inf_train_gen():
#     while True:
#         for images, target in train_gen():
#             # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
#             yield images
# gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

train_data_loader.setup('Training', randomized=True, verbose=True)
for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    
    for i in range(CRITIC_ITERS): 
        print('Critic iterations: ', i)    
        try: _, _, batch = next(train_data_loader) 
        except: train_data_loader.setup('Training', randomized=True, verbose=True)

        _data = batch["Image"]
        if _data.shape[0] != BATCH_SIZE: continue

        # _data = gen.next()
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])


        if use_cuda:
            real_data = real_data.cuda(gpu)
        # real_data_v = autograd.Variable(real_data)
        real_data_v = real_data
        # print(real_data_v.requires_grad)
        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        D_real = netD(real_data_v)
        D_real_obj = -D_real.mean()
        D_real_obj.backward()

        # train with fake
        noise = torch.randn(BATCH_SIZE, 3, 32, 32)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = noise
        # noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        # fake = autograd.Variable(netG(noisev).data)
        fake = netG(noisev)

        inputv = fake
        D_fake = netD(inputv)
        D_fake_obj = D_fake.mean()
        # D_fake_obj.backward()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v, fake)
        total_D_obj = D_fake_obj + gradient_penalty
        # gradient_penalty.backward()
        total_D_obj.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake


        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    print('Generator iteration: ', iteration)
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    # noisev = autograd.Variable(noise)
    noisev = noise
    fake = netG(noisev)
    G = netD(fake)
    G_obj = -G.mean()
    G_obj.backward()

    G_cost = G_obj
    optimizerG.step()

    # # Write logs and save samples
    # lib.plot.plot('./tmp/cifar10/train disc cost', D_cost.cpu().data.numpy())
    # lib.plot.plot('./tmp/cifar10/time', time.time() - start_time)
    # lib.plot.plot('./tmp/cifar10/train gen cost', G_cost.cpu().data.numpy())
    # lib.plot.plot('./tmp/cifar10/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # # Calculate inception score every 1K iters
    # if False and iteration % 1000 == 999:
    #     inception_score = get_inception_score(netG)
    #     lib.plot.plot('./tmp/cifar10/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 500 == 0:
        # dev_disc_costs = []
        # for images, _ in dev_gen():
        #     images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        #     imgs = torch.stack([preprocess(item) for item in images])

        #     # imgs = preprocess(images)
        #     if use_cuda:
        #         imgs = imgs.cuda(gpu)
        #     imgs_v = autograd.Variable(imgs, volatile=True)

        #     D = netD(imgs_v)
        #     _dev_disc_cost = -D.mean().cpu().data.numpy()
        #     dev_disc_costs.append(_dev_disc_cost)
        # # lib.plot.plot('./tmp/cifar10/dev disc cost', np.mean(dev_disc_costs))

        generate_image(iteration, netG)

    # # Save logs every 100 iters
    # if (iteration < 5) or (iteration % 100 == 99):
    #     lib.plot.flush()
    # lib.plot.tick()
