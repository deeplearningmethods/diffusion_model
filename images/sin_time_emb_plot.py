# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:02:19 2023

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import torch

def SinusoidalPositionEmbeddings(dim, time):
        half_dim = dim // 2
        time = torch.tensor([time])
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
time=1000
dim = 64
matrix = torch.zeros((dim, time))
for i in range(time):
  matrix[:,i] = SinusoidalPositionEmbeddings(dim, i)
# Plot the sinusoidal time embedding

# Set the figure size
plt.figure(figsize=(6, 4))

#plt.title('Sinusoidal Position Embeddings')
plt.xlabel('Time step')
plt.ylabel('Embedding dimension')

im = plt.imshow(matrix, cmap='viridis', aspect='auto')
cbar = plt.colorbar(im, )
