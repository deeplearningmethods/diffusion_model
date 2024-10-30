# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:30:04 2024

@author: dgallon
"""


import matplotlib.pyplot as plt
import numpy as np
import math
import time

alpha1= 1- 10**(-4)
alphaT= 0.98
T = 1000

alphas = np.linspace(alpha1,alphaT,T)

bar_alphas = np.cumprod(alphas)
    
label = r'$\tilde{\alpha}_t$'


plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True
})

plt.plot(range(1,T+1), bar_alphas)
plt.ylabel(label)
plt.xlabel('Diffusion step $t$')
#plt.title('k')
plt.tight_layout()
plt.savefig("alphabar.pdf", bbox_inches="tight")
