# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:26:39 2023

@author: Davide
"""
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L272 ispiration for DDPM DDIM
# https://github.com/gmongaras/Diffusion_models_from_scratch/blob  inspiration for classifier free guidance

from unet import Unet
from diffusion_model import GaussianDiffusion
from trainer import Trainer
from dataset import Dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")

# Select the classes we want to use in the model and the desired image size. The list of selected classes must contain the exact name of each desired class
selected_class = [i for i in range(100)]
img_size = 64
# initialize the dataset, we can choose among 'FGVCAIRCRAFT','Flowers102','StanfordCars','MNIST', 'CIFAR10'
dataset = Dataset('Flowers102', (img_size,img_size), selected_class = selected_class, grayscale = False)

# show examples of images in the dataset
dataset.show_images()

# return the dataset
data = dataset.get_data()


# initialize the unet
model = Unet(    in_channels=3,            # channel of input images
                  init_conv_dim=64,         # initial channel dimension of Unet
                  time_dim=64,              # time dimension after sin embedding
                  context_dim = 64,         # context inner dimension 
                  dim_factor = (1, 2, 4),   # scale factors of each Unet level
                  img_size=img_size,        # size of input images
                  num_classes = len(selected_class),  # number of classes
                  resnet_block_groups = 8,   # size of block groups
                  learned_variance = False,  # needed to pass from DDPM to improved DDPM (if True remember that the output dimension doubles)
                  attn_dim_head = 16,        # attention dim head
                  attn_heads = 4,            # attention head (occhio che deve tornare dim_head*head)
                  )

# The load previously trained model (if exists) 
#model = torch.load('modello')

# Initiate diffusion model
diffusion = GaussianDiffusion(
    model,                          # unet
    device=device,                  # device where the computation are done
    beta_schedule = 'linear',       # beta scheduler, can be linear or cosine
    schedule_fn_kwargs = {'beta_start':0.0001, 'beta_end':0.002}, # CIFAR10 and colors {'beta_start':0.0001, 'beta_end':0.0008} no colors {'beta_start':0.00001, 'beta_end':0.0007}
    ddim_sampling_eta = 1.,         # if 1 then we have a DDPM, if 0 we have DDIM. Any number between 0 and 1 means an interpolation between DDPM and DDIM
    w = 2,                          # constant used in sampling phase for classifier free guidance. In [0,10], 0 means no classifier free guidance.
    p_uncond = 0.2,                 # Probability of training on a null class for classifier free guidance. In [0,1), usually 0.1 or 0.2
    )

# Simulate a forward diffusion, we plot a image in our dataset adding gradually some noise 
diffusion.simulate_forward_diffusion(data[0][0], hist = True)
#diffusion.sample_plot_image()

# initialize the trainer, we can also specify train batch size, epochs, ...
trainer = Trainer(diffusion,
                  data,
                  device,
                  train_batch_size = 80,
                  epochs=40)

# It trains the model, every two epochs it plots a backward diffusion and saves the model
trainer.train()

# Generate an image after the training, one can specify
# - desired_class: the index of the class, e.g. 0,1,... 
# - custom_entry: specify directly a custom ohe vector
img_final = diffusion.sample_plot_image(epoch = 'final', desired_class = None, custom_entry = None)
diffusion.sample_plot_image_grid(epoch = 'final', desired_class = None, custom_entry = None)
