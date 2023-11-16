# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:03:40 2021

@author: Liangchao Zhu
"""
import numpy as np

def arrangeIndex():
    import h5py
    # matFile = 'nodIdx_matrix7.mat'
    matFile = 'nodIdx_matrix7_notPeriodic.mat'
    matData = h5py.File(matFile,'r')
    print(list(matData.items()))
    nodIdx = np.transpose( matData['nodIdx'][()])
    nodIdx = nodIdx.astype(np.int)
    print(nodIdx.shape,nodIdx.dtype,nodIdx.min(),nodIdx.max())
    
    from nodidxMapping import nodidxMapping
    _nodIdxMapping = nodidxMapping(40)
    print(_nodIdxMapping.shape, _nodIdxMapping.dtype, _nodIdxMapping.min(), _nodIdxMapping.max())
    
    nodIdx_tensor = np.zeros((41,41,41,27,3),dtype = np.int)
    for i in range(nodIdx.shape[0]):
        ii =  _nodIdxMapping[i,0]
        jj =  _nodIdxMapping[i,1]
        kk =  _nodIdxMapping[i,2]
        for j in range(27):
            nodIdx_ij = nodIdx[i,j]
            if nodIdx_ij<68921:
                nodIdx_tensor[ii,jj,kk,j,:] = _nodIdxMapping[nodIdx_ij,:]
            else:
                nodIdx_tensor[ii,jj,kk,j,:] = np.array([-1,-1,-1])
    return nodIdx_tensor
if __name__ == "__main__":
    nodIdx_tensor = arrangeIndex()