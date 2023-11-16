# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 22:22:39 2021

@author: Liangchao Zhu
"""
import numpy as np
import torch
import torch.nn.functional as F
def kernelForType(datatype = np.float32):
    filterKernel = np.zeros((2,2,2),dtype = datatype)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                Eleindex = i + 2*j + 4*k
                filterKernel[i,j,k] = 2**Eleindex
    return filterKernel
def typeH8(rho):
    if rho.dtype == torch.double:
        datatype = np.float64
    else:
        datatype = np.float32
    filterKernel = torch.from_numpy(kernelForType(datatype)[np.newaxis,np.newaxis]).to(rho.device)
    # print(filterKernel.dtype)
    H8Types = F.conv3d(rho,filterKernel,padding = 1)
    return H8Types.int()
def typeH8_(rho,filterKernel):
    H8Types = F.conv3d(rho,filterKernel,padding = 1)
    return H8Types.int()
if __name__ == "__main__":
    rho = torch.ones((8,1,40,40,40),dtype = torch.float32)
    rho = torch.ones((8,1,4,4,4),dtype = torch.float32)
    print(rho.shape)
    H8Types = typeH8(rho)