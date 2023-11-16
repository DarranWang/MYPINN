from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import time
#import feconv_cuda


def data_pre(batchsize = 512):
    print(f'**** batchsize = {batchsize} ****')
    # U = torch.rand((batchsize,18,40,40,40),dtype = torch.float)
    U = torch.ones((batchsize,18,40,40,40),dtype = torch.float)
    # rho = torch.rand((batchsize,1,40,40,40),dtype = torch.float)
    rho = torch.ones((batchsize,1,40,40,40),dtype = torch.float)
    rho[rho<=0.5] = 0
    rho[rho >0.5] = 1
    print('discre rho:', torch.abs((rho-1)*rho).max())
    
    resolution = 40 
    E = 1; nu = 0.3
    from ConstMatricesForHomogenization import ISOelasticitytensor,LocalKeFe
    D0 = ISOelasticitytensor(E, nu)
    Ke,Fe,intB = LocalKeFe(resolution,D0)
    h = 1.0/resolution
    nele = resolution**3
    I = np.eye(6)
    datashape = resolution
    # Ke2 = np.loadtxt("3D/k.txt", delimiter=',', dtype=np.float64) /2
    # Fe2 = np.loadtxt("3D/f.txt", delimiter=',', dtype=np.float64) /4
    # Ke = torch.from_numpy(Ke).to(device)
    # Fe = torch.from_numpy(Fe).to(device)
    
    from PeriodicMesh3D import PeriodicMesh3D,edofMatrix
    eleidx,MESH,V = PeriodicMesh3D(resolution)
    # mesh = np.loadtxt("3D/40mesh.txt", delimiter='\t', dtype=np.int)
    edofMat = edofMatrix(MESH)
    
    return U,rho,Ke,edofMat
def originalMethod_check(output_img,input,Ke, edofMat):
    
    size = input.shape[0]
    # 3d rho的顺序?
    pp = input.view(size, -1, 1, 1)#.to(device)
    K = pp * Ke  # [bs, 8000, 24, 24]
    # F = pp * Fe # [bs, 8000, 24, 6]
    
    ref18 = output_img.contiguous().view(size,18,-1)
    # map0 = ref18[:,0::6].permute((0,2,1)).contiguous().view(size,-1,1)
    # map1 = ref18[:,1::6].permute((0,2,1)).contiguous().view(size,-1,1)
    # map2 = ref18[:,2::6].permute((0,2,1)).contiguous().view(size,-1,1)
    # map3 = ref18[:,3::6].permute((0,2,1)).contiguous().view(size,-1,1)
    # map4 = ref18[:,4::6].permute((0,2,1)).contiguous().view(size,-1,1)
    # map5 = ref18[:,5::6].permute((0,2,1)).contiguous().view(size,-1,1)
    
    map0 = ref18[:,0:3].permute((0,2,1)).contiguous().view(size,-1,1)
    map1 = ref18[:,3:6].permute((0,2,1)).contiguous().view(size,-1,1)
    map2 = ref18[:,6:9].permute((0,2,1)).contiguous().view(size,-1,1)
    map3 = ref18[:,9:12].permute((0,2,1)).contiguous().view(size,-1,1)
    map4 = ref18[:,12:15].permute((0,2,1)).contiguous().view(size,-1,1)
    map5 = ref18[:,15:18].permute((0,2,1)).contiguous().view(size,-1,1)
    
    # ref18 = output_img.permute((0,2,3,4,1)).contiguous().view(size,-1,18)
    # map0 = ref18[:,:,0:3].contiguous().view(size,-1,1)
    # map1 = ref18[:,:,3:6].contiguous().view(size,-1,1)
    # map2 = ref18[:,:,6:9].contiguous().view(size,-1,1)
    # map3 = ref18[:,:,9:12].contiguous().view(size,-1,1)
    # map4 = ref18[:,:,12:15].contiguous().view(size,-1,1)
    # map5 = ref18[:,:,15:18].contiguous().view(size,-1,1)
    ref_map = torch.cat([map0,map1,map2,map3,map4,map5], 2)# [bs,3*40**3,6]
    # ref_map[:,0:3,:] = 0

    # U = torch.zeros([size, 8000, 24, 6]).to(device)
    U = ref_map[:, edofMat, :]

    UT = U.permute([0, 1, 3, 2])
    # losst1 = torch.matmul(torch.matmul(UT, K), U).sum()
    # print(losst1)
    
    # FU = (U * F).sum()

    # losst1 = 
    UKU = torch.matmul(torch.matmul(UT, K), U)
    UKU0 = UKU[:,:,0,0].sum()
    UKU1 = UKU[:,:,1,1].sum()
    UKU2 = UKU[:,:,2,2].sum()
    UKU3 = UKU[:,:,3,3].sum()
    UKU4 = UKU[:,:,4,4].sum()
    UKU5 = UKU[:,:,5,5].sum()

    losst1 = UKU0+UKU1+UKU2+UKU3+UKU4+UKU5
    print(losst1)
    return losst1
    
def feconv_check(U,rho,Ke):
    from feconv import FECONV
    from feconv import FEconvFunction
    
    print('FECONV imported')
    
    # batchsize = 512
    # print(f'**** batchsize = {batchsize} ****')
    from periodicU import periodicU
    U = periodicU(U)
    
    from getTypeH8 import typeH8
    H8types = typeH8(rho)
    H8types = H8types.int()

    from arrangeIndex import arrangeIndex
    nodIdx = arrangeIndex()

    # Ke = np.eye(24).astype(np.float32)
    from symbolicExec_vec import getFilters
    filters = getFilters(Ke)
    filters = filters.astype(np.float32)
    
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)

    nodIdx = nodIdx.astype(np.int32)
    U = U.to(device)
    # H8types = torch.from_numpy(H8types).to(device)
    H8types = H8types.to(device)
    nodIdx = torch.from_numpy(nodIdx).to(device)
    filters = torch.from_numpy(filters).to(device)
    filters = filters*1e6
    print("INPUT info.:----------------------------------")
    print('* U	:',U.cpu().numpy().shape,U.dtype,U.sum().cpu().numpy())
    print('* H8types	:',H8types.cpu().numpy().shape,H8types.dtype,H8types.sum().cpu().numpy(),H8types.min().cpu().numpy(),H8types.max().cpu().numpy())
    print('* nodIdx	:',nodIdx.cpu().numpy().shape,nodIdx.dtype,nodIdx.sum().cpu().numpy(),nodIdx.min().cpu().numpy(),nodIdx.max().cpu().numpy())
    print('* filters	:',filters.cpu().numpy().shape,filters.dtype,filters.sum().cpu().numpy())
    
    steps = 10
    convOP = FECONV().to(device)
    
    start = time.perf_counter()
    # for i in range(steps):
    KU = convOP(U,H8types,nodIdx,filters)
    # KU = FEconvFunction.apply(U,H8types,nodIdx,filters)
    uku = (KU*U).sum((2,3,4))
    uku1 = uku.view(-1,6,3).sum((2))
    elapsed = time.perf_counter() - start
    
    print(f"elapsed in {elapsed} s")
    print("OUTPUT info.:---------------------------------")
    print('* KU	:',KU.shape,KU.dtype,KU.device)
    print('* U 	:',U.shape,U.dtype,U.device)
    #uku2 = uku.view(-1,3,6).sum((1))
    print('* UKU	:',uku1.shape,uku1.dtype,uku1.device)
    print(uku1.cpu().numpy()[0,0])
    sum0 = uku[0,:3].sum()
    print(uku1.shape, uku1[0,0]-sum0)
    print(uku1[:2])

def oricheck(U,rho,Ke,edofMat):
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)
    edofMat = torch.from_numpy(edofMat).to(device).long()
    Ke = torch.from_numpy(Ke).to(device)
    rho = rho.to(device)
    U = U.to(device)
    
    start = time.perf_counter()
    originalMethod_check(U,rho,Ke, edofMat)
    elapsed = time.perf_counter() - start
    print(f"elapsed in {elapsed} s")

if __name__ == "__main__":
    print('modify mark 1')
    U,rho,Ke,edofMat = data_pre(batchsize = 32)
    feconv_check(U,rho,Ke)
    
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # from periodicU import periodicU
    # U = periodicU(U)
    # U = U.to(device)
    # tmp = U*U
    # print('tmp: ',tmp.shape,tmp.device)
    
    
    #oricheck(U,rho,Ke,edofMat)
    
