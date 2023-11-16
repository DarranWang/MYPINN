# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:36:29 2021

@author: Liangchao Zhu
"""

import numpy as np

def nodidxMapping(num):
   nele = num**3
   nnod = (num+1)**3
   nodenrs = np.arange(nnod).reshape((num+1,num+1,num+1))
   # edofvec = nodenrs[:-1,:-1,:-1].reshape((nele,1))
   # edofMat = np.tile(edofvec,(1,8))+np.tile(np.array([0,1,num+1,num+2,(num+1)**2,(num+1)**2+1,(num+1)**2+num+1,(num+1)**2+num+2]),(nele,1))
   # # print(nodenrs[0])
   # # print(edofvec)
   # # print(edofMat)
   # nnpArray=np.arange(nele).reshape(num,num,num);
   # eleidx = nnpArray
   nodIdx = np.zeros((nnod,3),dtype = np.int)
   for i in range(num+1):
       for j in range(num+1):
           for k in range(num+1):
               nodIdx[nodenrs[i,j,k],0]=k
               nodIdx[nodenrs[i,j,k],1]=j
               nodIdx[nodenrs[i,j,k],2]=i
   return nodIdx

if __name__ == "__main__":
    nodIdx = nodidxMapping(2)