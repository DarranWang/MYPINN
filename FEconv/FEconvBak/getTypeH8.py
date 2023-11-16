# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 22:22:39 2021

@author: Liangchao Zhu
"""
import numpy as np
import torch
import torch.nn.functional as F
def kernelForType():
    filterKernel = np.zeros((2,2,2),dtype = np.float32)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                Eleindex = i + 2*j + 4*k
                filterKernel[i,j,k] = 2**Eleindex
    return filterKernel
def typeH8(rho):
    filterKernel = torch.from_numpy(kernelForType()[np.newaxis,np.newaxis])
    H8Types = F.conv3d(rho,filterKernel,padding = 1)
    return H8Types
if __name__ == "__main__":
    rho = torch.ones((8,1,40,40,40),dtype = torch.float32)
    print(rho.shape)
    H8Types = typeH8(rho)