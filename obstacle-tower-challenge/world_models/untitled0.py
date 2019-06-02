#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 01:24:01 2019

@author: mostafa
"""

import numpy as np
import torch
from torch.distributions import Normal

def gaussian_pdf(x, mu, sigmasq):
    res = (1/torch.sqrt(2*np.pi*sigmasq)) 
    res = res * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2)
    return res

x = torch.tensor([[-10.0, 1.0]])
mu = torch.tensor([[0.0]])
sig = torch.tensor([[0.1]])

#res1 = gaussian_pdf(x, mu, sig)
#print(res1)


normal_dist = Normal(mu, sig)
#res2 = normal_dist.cdf(x)
#print(res2)

res3 = normal_dist.log_prob(x)
print(res3)

