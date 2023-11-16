# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 23:14:53 2021

@author: Liangchao Zhu
"""
import torch
def periodicU(U):
    # [batchsize,18,40,40,40]
    datashape = U.shape[2]
    # pU = torch.zeros((U.shape[0],18,datashape+1,datashape+1,datashape+1),device = U.device)
    U = torch.cat((U,U[:,:,[0]]),2)
    U = torch.cat((U,U[:,:,:,[0]]),3)
    U = torch.cat((U,U[:,:,:,:,[0]]),4)
    return U

if __name__ == "__main__":
    # U = torch.rand((2,18,40,40,40))
    U = torch.rand((2,18,2,2,2))
    datashape = U.shape[2]
    pU = periodicU(U)
    print(torch.abs(pU[:,:,0]-pU[:,:,datashape]).sum())
    print(torch.abs(pU[:,:,:,0]-pU[:,:,:,datashape]).sum())
    print(torch.abs(pU[:,:,:,:,0]-pU[:,:,:,:,datashape]).sum())