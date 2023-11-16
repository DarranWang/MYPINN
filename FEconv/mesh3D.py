# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:56:31 2021

@author: Liangchao Zhu
"""

import numpy as np
def mesh3D(num):
   nele = num**3
   nnod = (num+1)**3
   nodenrs = np.arange(nnod).reshape((num+1,num+1,num+1))
   edofvec = nodenrs[:-1,:-1,:-1].reshape((nele,1))
   edofMat = np.tile(edofvec,(1,8))+np.tile(np.array([0,1,num+1,num+2,(num+1)**2,(num+1)**2+1,(num+1)**2+num+1,(num+1)**2+num+2]),(nele,1))
   # print(nodenrs[0])
   # print(edofvec)
   # print(edofMat)
   # nnpArray=np.arange(nele).reshape(num,num,num);
   nnpArray=np.arange(nnod).reshape(num+1,num+1,num+1);
   eleidx = nnpArray
   
   
   # tmp = np.zeros((num+1,num,num))
   # tmp[:num,:,:] = nnpArray
   # tmp[-1,:,:] = nnpArray[0,:,:]
   
   # tmp2 = np.zeros((num+1,num+1,num))
   # tmp2[:,:num,:] = tmp
   # tmp2[:,-1,:] = tmp[:,0,:]
   
   # tmp3 = np.zeros((num+1,num+1,num+1))
   # tmp3[:,:,:num] = tmp2
   # tmp3[:,:,-1] = tmp2[:,:,0]
   # nnpArray = tmp3
   
   # print(nnpArray)
   # dofvector = np.zeros((nod,1))
   dofvector = nnpArray.flatten()
   # print(dofvector)
   mesh=dofvector[edofMat]
   # print(mesh)
   h = 1.0/num
   VE = np.zeros((nnod,3))
   for i in range(num+1):
       for j in range(num+1):
           for k in range(num+1):
               VE[eleidx[k,j,i],:] = np.array([i*h,j*h,k*h])
   MESH=mesh
   mesh_ = MESH.copy().astype(np.int)
   # mesh_[:,3] = MESH[:,2]
   # mesh_[:,2] = MESH[:,3]
   # mesh_[:,7] = MESH[:,6]
   # mesh_[:,6] = MESH[:,7]
   return eleidx,mesh_,VE
def edofMatrix(MESH):
    tmp1 = 3*MESH
    tmp2 = 3*MESH+1
    tmp3 = 3*MESH+2
    edofMat = np.zeros([MESH.shape[0], 24], dtype=np.int)
    edofMat[:,::3] = tmp1
    edofMat[:,1::3] = tmp2
    edofMat[:,2::3] = tmp3
    return edofMat

if __name__ == "__main__":
    eleidx,mesh_, coor = Mesh3D(2)