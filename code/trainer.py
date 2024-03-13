# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:50:17 2023

@author: Davide
"""

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import Adam
import torch
import numpy as np

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        device,
        *,
        train_batch_size = 16,              # batch size
        gradient_accumulate_every = 1,      # not yet used
        train_lr = 1e-4,                    # learning rate
        epochs =60,                         # number of epochs
        mixed_precision_type = 'fp16',      # not yet used
        max_grad_norm = 1.,                 # not yet used
    ):
        super().__init__()

        # model

        self.diff_model = diffusion_model.to(device)
        self.in_channels = diffusion_model.in_channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling # modify code
        
        # device
        self.device = device
        
        # dataset and dataloader
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, drop_last=True ) # we can add number of workers
       
        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)      
        
        # training parameters
        self.step = 0
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        assert self.diff_model.model.context_dim is not None or self.diff_model.p_uncond>0, 'To use classifier free guidance we need class information!'
        
            
    def train(self,save=True):
            """
            Train the model, when save=True it saves model parameter. every 2 epochs plot an reverse diffusion process.
            """

            for epoch in tqdm(range(self.epochs)):
                
                loss_vec = []
                
                for step, batch in enumerate(self.dl):

                   
                    self.opt.zero_grad()
                    
                    t = torch.randint(0, self.diff_model.num_timesteps , (self.train_batch_size,), device=self.device).long()
                    context = torch.nn.functional.one_hot(batch[1], num_classes=self.diff_model.model.num_classes ).to(self.device).float() #if self.use_cond else None
                    
                    if self.diff_model.p_uncond>0:
                        prob = torch.rand(batch[0].shape[0])
                        mask = prob < self.diff_model.p_uncond
                        context[mask] = 0
                    
                    loss = self.diff_model(batch[0].to(self.device), t, context)
                    loss_vec.append(loss.detach().cpu().numpy())
                    loss.backward()
                    self.opt.step()
                    if epoch % 2 == 0 and step == 0:
                        #print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                        self.diff_model.sample_plot_image(epoch)


                print(f"Epoch {epoch} |  Loss: {np.mean(loss_vec)} ")

            if save == True:
                          torch.save(self.diff_model.model, 'modello')
                
   
