from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import time
#import feconv_cuda

#%% data_pre
def data_pre_dtype(batchsize = 1,resolution = 40,isFloat64=False):
    Ke,Fe,edofMat = data_pre(batchsize,resolution)
    if isFloat64:
        Ke = Ke.astype(np.float64)
        Fe = Fe.astype(np.float64)
    return Ke,Fe,edofMat

def data_pre(batchsize = 1,resolution = 40):

    E = 1e6; nu = 0.3
    from ConstMatricesForHomogenization import ISOelasticitytensor,LocalKeFe
    D0 = ISOelasticitytensor(E, nu)
    Ke,Fe,intB = LocalKeFe(resolution,D0)
    # h = 1.0/resolution
    # nele = resolution**3
    # I = np.eye(6)
    # datashape = resolution
    
    from PeriodicMesh3D import PeriodicMesh3D,edofMatrix
    eleidx,MESH,V = PeriodicMesh3D(resolution)
    # from mesh3D import mesh3D,edofMatrix
    # eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    
    print('====================== data prepared for homogenization')
    # Ke = np.ones(Ke.shape,dtype = Ke.dtype)
    # Ke = np.eye(24,dtype = Ke.dtype)
    # Ke[0,1] = 1
    return Ke,Fe,edofMat
#=============================================================================

#%% FEconv_cuda
def feconvNet_periodicU_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz):
    
    Up,H8types,nodIdx,filters = datapre_feconv(U,rho,Ke)
    
    
    from feconv import FEconvNet_periodicU_H8types
    
    print('FECONV imported')

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)

    # U = U.to(device)
    H8types = H8types.to(device)
    rho = rho.to(device)
    nodIdx =   nodIdx.to(device)
    filters = filters.to(device)
    # print("INPUT info.:----------------------------------")
    # print('* U	:',U.cpu().numpy().shape,U.dtype,U.sum().cpu().numpy())
    # print('* H8types	:',H8types.cpu().numpy().shape,H8types.dtype,H8types.sum().cpu().numpy(),H8types.min().cpu().numpy(),H8types.max().cpu().numpy())
    # print('* nodIdx	:',nodIdx.cpu().numpy().shape,nodIdx.dtype,nodIdx.sum().cpu().numpy(),nodIdx.min().cpu().numpy(),nodIdx.max().cpu().numpy())
    # print('* filters	:',filters.cpu().numpy().shape,filters.dtype,abs(filters).sum().cpu().numpy())
    
    steps = 1
    convOP = FEconvNet_periodicU_H8types(Ke).to(device)

    start = time.perf_counter()
    for i in range(steps):
        # KU = convOP(U,rho,nodIdx,filters)
        KU = convOP(U,H8types)
        # KU = FEconvFunction.apply(U,H8types,nodIdx,filters)
        uku = (KU*U).sum((2,3,4))
        uku1 = uku.view(-1,6,3).sum((2))
    elapsed = time.perf_counter() - start
    
    print(f"{steps} steps, {elapsed/steps} s/step")
    print("OUTPUT info.:---------------------------------")
    print('* KU	:',KU.shape,KU.dtype,KU.device)
    print('* U 	:',U.shape,U.dtype,U.device)
    #uku2 = uku.view(-1,3,6).sum((1))
    print('* UKU	:',uku1.shape,uku1.dtype,uku1.device)
    #print(uku1.cpu().numpy()[0,0])
    #sum0 = uku[0,:3].sum()
    #print(uku1.shape, uku1[0,0]-sum0)
    print(uku1.shape,"* ",uku1.cpu().detach().numpy())

    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 2; Idxz =3;
    # FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    # print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    print(f'U[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',U[ibatch,outidx,Idxx,Idxy,Idxz])
    
    # print("******************** Gradients")
    # L = uku1.sum()
    # print("L = ",L)
    # L.backward()
    # gradU = U_40.grad()
    # print(f"gradU.shape = {gradU.shape}")
    return uku1.sum()

def FEconv_runTwice(U,rho,Ke):
    from feconv import FEconvNet_periodicU_H8types
    convOP = FEconvNet_periodicU_H8types(Ke).to(device)
    from getTypeH8 import typeH8
    from periodicU import periodicU
    
    
    start = time.perf_counter()
    H8types = typeH8(rho).to(device)
    elapsed = time.perf_counter() - start
    print(f"preprocess rho: {elapsed} s")
    
    start = time.perf_counter()
    U = periodicU(U).to(device)
    elapsed = time.perf_counter() - start
    print(f"preprocess U: {elapsed} s")
    
    
    steps = 10
    for i in range(10):
        start = time.perf_counter()
        KU = convOP(U,H8types)
        elapsed = time.perf_counter() - start
        print(f"Round {i+1}: {elapsed} s, KU.sum() = {KU.sum()}")
    
    # start = time.perf_counter()
    # KU = convOP(U,H8types)
    # elapsed = time.perf_counter() - start
    # print(f"2nd round: {elapsed} s")
    # print(KU.sum())
    # return KU
    
def feconvNet_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz):
    
    U_bak,H8types,nodIdx,filters = datapre_feconv(U,rho,Ke)
    
    print("======================= FEconvNet--------------")
    from feconv import FEconvNet
    
    print('FECONV imported')

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)

    # U = U.to(device)
    # H8types = H8types.to(device)
    rho = rho.to(device)
    nodIdx =   nodIdx.to(device)
    filters = filters.to(device)
    # print("INPUT info.:----------------------------------")
    # print('* U	:',U.cpu().numpy().shape,U.dtype,U.sum().cpu().numpy())
    # print('* H8types	:',H8types.cpu().numpy().shape,H8types.dtype,H8types.sum().cpu().numpy(),H8types.min().cpu().numpy(),H8types.max().cpu().numpy())
    # print('* nodIdx	:',nodIdx.cpu().numpy().shape,nodIdx.dtype,nodIdx.sum().cpu().numpy(),nodIdx.min().cpu().numpy(),nodIdx.max().cpu().numpy())
    # print('* filters	:',filters.cpu().numpy().shape,filters.dtype,abs(filters).sum().cpu().numpy())
    
    steps = 1
    convOP = FEconvNet().to(device)
    
    U_40 = U
    start = time.perf_counter()
    for i in range(steps):
        KU,U = convOP(U_40,rho,nodIdx,filters)
        # KU = FEconvFunction.apply(U,H8types,nodIdx,filters)
        uku = (KU*U).sum((2,3,4))
        uku1 = uku.view(-1,6,3).sum((2))
    elapsed = time.perf_counter() - start
    
    print(f"{steps} steps, {elapsed/steps} s/step")
    print("OUTPUT info.:---------------------------------")
    print('* KU	:',KU.shape,KU.dtype,KU.device)
    print('* U 	:',U.shape,U.dtype,U.device)
    #uku2 = uku.view(-1,3,6).sum((1))
    print('* UKU	:',uku1.shape,uku1.dtype,uku1.device)
    #print(uku1.cpu().numpy()[0,0])
    #sum0 = uku[0,:3].sum()
    #print(uku1.shape, uku1[0,0]-sum0)
    print(uku1.cpu().detach().numpy())
    
    
    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 2; Idxz =3;
    # FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    # print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    print(f'U[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',U[ibatch,outidx,Idxx,Idxy,Idxz])
    
    # print("******************** Gradients")
    # L = uku1.sum()
    # print("L = ",L)
    # L.backward()
    # gradU = U_40.grad()
    # print(f"gradU.shape = {gradU.shape}")


def datapre_feconv(U,rho,Ke):
    from periodicU import periodicU
    U = periodicU(U)
    
    from getTypeH8 import typeH8
    H8types = typeH8(rho)
    H8types = H8types.int()

    from arrangeIndex import arrangeIndex
    nodIdx = arrangeIndex()
    # nodIdx = nodIdx.astype(np.int32)

    from symbolicExec_vec2 import getFilters
    filters = getFilters(Ke)
    # filters = filters.astype(np.float32)
    nodIdx = torch.from_numpy(nodIdx)
    filters = torch.from_numpy(filters)
    print('====================== data prepared for FEconv')
    print('*       U:',      U.shape,       U.dtype)
    print('* H8types:',H8types.shape, H8types.dtype)
    print('*  nodIdx:', nodIdx.shape,  nodIdx.dtype)
    print('* filters:',filters.shape, filters.dtype)
    return U,H8types,nodIdx,filters

def feconv_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz):
    
    U,H8types,nodIdx,filters = datapre_feconv(U,rho,Ke)
    
    from feconv import FECONV
    from feconv import FEconvFunction
    
    print('FECONV imported')

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)

    U = U.to(device)
    H8types = H8types.to(device)
    nodIdx =   nodIdx.to(device)
    filters = filters.to(device)
    # print("INPUT info.:----------------------------------")
    # print('* U	:',U.cpu().numpy().shape,U.dtype,U.sum().cpu().numpy())
    # print('* H8types	:',H8types.cpu().numpy().shape,H8types.dtype,H8types.sum().cpu().numpy(),H8types.min().cpu().numpy(),H8types.max().cpu().numpy())
    # print('* nodIdx	:',nodIdx.cpu().numpy().shape,nodIdx.dtype,nodIdx.sum().cpu().numpy(),nodIdx.min().cpu().numpy(),nodIdx.max().cpu().numpy())
    # print('* filters	:',filters.cpu().numpy().shape,filters.dtype,abs(filters).sum().cpu().numpy())
    
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
    #print(uku1.cpu().numpy()[0,0])
    #sum0 = uku[0,:3].sum()
    #print(uku1.shape, uku1[0,0]-sum0)
    print(uku1.cpu().numpy())
    
    
    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 2; Idxz =3;
    # FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    # print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])

#%% element-wise
def originalMethod_check(output_img,input,Ke, edofMat):
    
    input=input.permute((0,1,4,3,2))
    
    size = input.shape[0]
    # 3d rho的顺序?
    pp = input.contiguous().view(size, -1, 1, 1)#.to(device)
    K = pp * Ke  # [bs, 8000, 24, 24]
    # F = pp * Fe # [bs, 8000, 24, 6]
    
    #output_img = output_img.permute((0,1,4,3,2))
    ref_map = Utensor2vec(output_img)
    U = ref_map[:, edofMat, :]#[bs,40^3,24,6]

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
    print(np.array([UKU0.item(), UKU1.item(), UKU2.item(), 
                    UKU3.item(), UKU4.item(), UKU5.item()]))
    return losst1

def oricheck(U,rho,Ke,edofMat):
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)
    edofMat = torch.from_numpy(edofMat).to(device).long()
    Ke = torch.from_numpy(Ke).to(device)
    rho = rho.to(device)
    # U = U.to(device)
    
    print("INPUT info.:----------------------------------")
    print('* U	    :',U.cpu().detach().numpy().shape,U.dtype,U.sum().cpu().detach().numpy())
    print('* rho	:',rho.cpu().numpy().shape,rho.dtype,rho.sum().cpu().numpy(),rho.min().cpu().numpy(),rho.max().cpu().numpy())
    print('* edofMat	:',edofMat.cpu().numpy().shape,edofMat.dtype,edofMat.sum().cpu().numpy(),edofMat.min().cpu().numpy(),edofMat.max().cpu().numpy())
    print('* Ke	    :',Ke.cpu().numpy().shape,Ke.dtype,abs(Ke).sum().cpu().numpy())
    
    # resolution = U.shape[2]
    # U = torch.rand((1,18,resolution,resolution,resolution),dtype = torch.float64,device=device,requires_grad=True)
    # U = U.requires_grad_()
    
    start = time.perf_counter()
    losst1 = originalMethod_check(U,rho,Ke, edofMat)
    elapsed = time.perf_counter() - start
    print(f"elapsed in {elapsed} s")


    return losst1
#%% assemble
def assembleKU_periodic(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz):
    print('=============assembleKU-------Periodic B.C.---------------')
    x = rho[0,0].cpu().numpy()
    x = np.transpose(x,(2,1,0))
    resolution = x.shape[0]
    nele = resolution**3; ndof = nele*3
    K,F = AssembleGlobalKF(x,Ke,Fe,edofMat,ndof,nele)
    print(K.dtype,K.shape)
    ref_map = Utensor2vec(U)
    
    uku = np.zeros((6))
    for i in range(6):
        Uvec = ref_map[0,:,i].cpu().detach().numpy()
        uku[i] = Uvec @ (K@Uvec)
    print(uku)


def assembleKU_check(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz):
    from periodicU import periodicU
    U = periodicU(U)
    
    x = rho[0,0].cpu().numpy()
    # resolution = x.shape[0]
    # nele = resolution**3; ndof = nele#3*(resolution+1)**3
    
    x = np.transpose(x,(2,1,0))
    resolution = x.shape[0]
    nele = resolution**3; ndof = 3*(resolution+1)**3
    
    from mesh3D import mesh3D,edofMatrix
    eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    
    K,F = AssembleGlobalKF(x,Ke,Fe,edofMat,ndof,nele)
    print(K.dtype,K.shape)

    ref_map = Utensor2vec(U)
    
    uku = np.zeros((6))
    for i in range(6):
        Uvec = ref_map[0,:,i].cpu().detach().numpy()
        uku[i] = Uvec @ (K@Uvec)
    print(uku)
    
    imap = outidx//3
    
    Uvec = ref_map[0,:,[imap]].cpu().detach().numpy()
    KU = K@Uvec
    print(f"imap = {imap},KU.shape = {KU.shape}")
    resolution = resolution + 1
    inodidx = Idxz*resolution**2 + Idxy*resolution + Idxx
    idofidx = inodidx*3 + outidx % 3
    print(f"inodidx = {inodidx}, idofidx = {idofidx}")
    print('*** assembleKU_i = ',KU[idofidx])
    print('*** U_i = ',Uvec[idofidx])
    # Uvec = np.ones((K.shape[0],1))
    # print(Uvec[3:].T @ (K[3:,:][:,3:]@Uvec[3:]))
    # Uvec[:3] = 0
    # print(Uvec.T @ (K@Uvec))
def AssembleGlobalK(x,Ke,edofMat,ndof,nele):
    from scipy import sparse
    penal = 1;
    rho = x.flatten()#[:nele]
    
    Emin = 0#1e-9;     
    rho = np.maximum(rho,Emin)
    
    iK = np.kron(edofMat,np.ones((24,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,24))).flatten()
    sK = ((Ke.flatten()[np.newaxis]).T*(rho**penal)).flatten(order='F')
    K = sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    return K

def AssembleGlobalKF(x,Ke,Fe,edofMat,ndof,nele):
    from scipy import sparse
    penal = 1;
    rho = x.flatten()#[:nele]
    
    Emin = 0#1e-9;     
    rho = np.maximum(rho,Emin)
    
    iK = np.kron(edofMat,np.ones((24,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,24))).flatten()
    sK = ((Ke.flatten()[np.newaxis]).T*(rho**penal)).flatten(order='F')
    K = sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
      
    iF = np.kron(edofMat,np.ones((6,1))).flatten()
    jF = np.kron(np.repeat(np.array(range(6))[np.newaxis],nele,axis = 0),np.ones((1,24))).flatten()
    sF = ((Fe.flatten(order='F')[np.newaxis]).T*(rho**penal)).flatten(order='F')
    F = sparse.coo_matrix((sF, (iF, jF)), shape=(ndof, 6)).tocsc()
    return K,F
#%% others
def KUperEleComputation():
    resolution = 2
    from mesh3D import mesh3D,edofMatrix
    eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    idofidx = 39
    nele = edofMat.shape[0]
    for iele in range(nele):
        edof = edofMat[iele,:]
        for j in range(24):
            if edof[j]==idofidx:
                for k in range(24):
                    print(f"Ke[{j},{k}]*U[{edof[k]}] + ",end = ' ')
                # Ke[j,:]*U[edof]
                print(' ')

def Utensor2vec(U):
    if len(U.shape)==5:#[bs,18,resolution,resolution,resolution]
        
        size = U.shape[0]
        U = U.permute((0,1,4,3,2))
        if U.shape[1]==18:
            ref18 = U.contiguous().view(size,18,-1) #[bs,18,40**3]       
            permuteList = (0,2,1)
            map0 = ref18[:,0:3].permute(permuteList).contiguous().view(size,-1,1)
            map1 = ref18[:,3:6].permute(permuteList).contiguous().view(size,-1,1)
            map2 = ref18[:,6:9].permute(permuteList).contiguous().view(size,-1,1)
            map3 = ref18[:,9:12].permute(permuteList).contiguous().view(size,-1,1)
            map4 = ref18[:,12:15].permute(permuteList).contiguous().view(size,-1,1)
            map5 = ref18[:,15:18].permute(permuteList).contiguous().view(size,-1,1)
            ref_map = torch.cat([map0,map1,map2,map3,map4,map5], 2)# [bs,3*40**3,6]
        if U.shape[1]==3:#[bs,3,resolution,resolution,resolution]
            ref3 = U.contiguous().view(size,3,-1) #[bs,18,40**3]
            ref_map = ref3.permute((0,2,1)).contiguous().view(size,-1,1)# [bs,3*40**3,1]
    if len(U.shape)==4:#[3,resolution,resolution,resolution]
        size = 1
        ref3 = U.contiguous().view(size,3,-1) #[bs,18,40**3]
        ref_map = ref3.permute((0,2,1)).contiguous().view(size,-1,1)# [bs,3*40**3,1]
    return ref_map


def fixeddofs(resolution):
    node = []
    for i in [0,resolution]:
        for j in [0,resolution]:
            for k in [0,resolution]:
                nodeidx = i*resolution**2 + j*resolution + k
                node.append(nodeidx)
    fixed = []
    for i in node:
        for j in range(3):
            fixed.append(i*3+j)
    return fixed


def filtercheck(Ke):
    from symbolicExec_vec2 import getFilters
    filters = getFilters(Ke)
    from symbolicExec_vec2 import symbolicExec
    theta = symbolicExec(Ke,1,1,1,1,1,1,1,1)
    print("---Ke----filters----theta---")
    print(Ke.dtype,filters[255,].dtype,theta.dtype)
    print(Ke.sum(),filters[255,].sum(),theta.sum())
    print(Ke[21,0],filters[255,0,0,0],theta[0,0,0])
    
    for ix in range(3):
        for iy in range(3):
            idxx = list(np.arange(ix,24,3))
            idxy = list(np.arange(iy,24,3))
            print(f"ix={ix},iy={iy}: ",Ke[idxx,:][:,idxy].shape,
                  Ke[idxx,:][:,idxy].sum(),filters[255,ix,iy].sum(),theta[ix,iy].sum())
            
    filters = torch.from_numpy(filters)
    print(Ke.dtype,filters[255,].dtype,theta.dtype)
    print(Ke.sum(),filters[255,].sum(),theta.sum())
    print(Ke[21,0],filters[255,0,0,0],theta[0,0,0])
    for ix in range(3):
        for iy in range(3):
            idxx = list(np.arange(ix,24,3))
            idxy = list(np.arange(iy,24,3))
            print(f"ix={ix},iy={iy}: ",Ke[idxx,:][:,idxy].shape,
                  Ke[idxx,:][:,idxy].sum(),filters[255,ix,iy].sum(),theta[ix,iy].sum())
    filters = filters.float()
    theta = theta.astype(np.float32)
    Ke = Ke.astype(np.float32)
    print(Ke.dtype,filters[255,].dtype,theta.dtype)
    for ix in range(3):
        for iy in range(3):
            idxx = list(np.arange(ix,24,3))
            idxy = list(np.arange(iy,24,3))
            print(f"ix={ix},iy={iy}: ",Ke[idxx,:][:,idxy].shape,
                  Ke[idxx,:][:,idxy].sum(),filters[255,ix,iy].sum(),theta[ix,iy].sum())


#%% FEconv_Pycheck
def FEconv_Pycheck_varU(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz):
    
    x = rho[0,0].cpu().numpy()
    resolution = x.shape[0]
    nele = resolution**3; ndof = 3*(resolution+1)**3
    
    
    from mesh3D import mesh3D,edofMatrix
    eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    K = AssembleGlobalK(x,Ke,edofMat,ndof,nele)
    
    inodidx = Idxx*(resolution+1)**2 + Idxy*(resolution+1) + Idxz
    idofidx = inodidx*3 + outidx % 3
    print(f"outidx,Idxx,Idxy,Idxz,inodidx,idofidx = {outidx,Idxx,Idxy,Idxz,inodidx,idofidx}")
    
    U,H8types,nodIdx,filters = datapre_feconv(U,rho,Ke)
    for i in range(3):
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    U2 = U
                    U2[0,i,ix,iy,iz]=2
                    print(f"(i,ix,iy,iz) = {i,ix,iy,iz}")
                    
                    
                    ref_map = Utensor2vec(U2)
                    Uvec = ref_map[ibatch,:,outidx // 3]
                    Uvec = Uvec.numpy()
                    KU = K@Uvec
                    print(f', assembleKU_[{idofidx}] = ',KU[idofidx],end='')
                    
                    convresult = FEconv_PyCheck_st(U2,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
def FEconv_PyCheck(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz):
    U,H8types,nodIdx,filters = datapre_feconv(U,rho,Ke)
    
    '''
    UKU = 0
    for outidx in range(3):
        for Idxx in range(40):
            for Idxy in range(40):
                for Idxz in range(40):
                    convresult = FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
                    UKU += convresult
    print('ConvUKU = ',UKU)
    '''
    convresult = FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    
    
    # for i in range(8):
    #     h8type = 2**i
    #     convresult = FEconv_PyCheck_st2(U,h8type,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    
    x = rho[0,0].cpu().numpy()
    x = np.transpose(x,(2,1,0))
    resolution = x.shape[0]
    nele = resolution**3; ndof = 3*(resolution+1)**3
    
    from mesh3D import mesh3D,edofMatrix
    eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    K = AssembleGlobalK(x,Ke,edofMat,ndof,nele)
    # print('global K: ',K.dtype,K.shape)
    
    
    ref_map = Utensor2vec(U)
    Uvec = ref_map[ibatch,:,outidx // 3]
    Uvec = Uvec.numpy()
    # Uvec = np.ones((K.shape[0]),dtype = K.dtype)
    
    KU = K@Uvec
    
    
    # inodidx = Idxx*(resolution+1)**2 + Idxy*(resolution+1) + Idxz
    inodidx = Idxz*(resolution+1)**2 + Idxy*(resolution+1) + Idxx
    idofidx = inodidx*3 + outidx % 3
    print(f"outidx,Idxx,Idxy,Idxz,inodidx,idofidx = {outidx,Idxx,Idxy,Idxz,inodidx,idofidx}")
    print(f'*** assembleKU_[{idofidx}] = ',KU[idofidx])
    # print(KU.T)
    # ndofs = K.shape[0]
    # dofs=np.arange(ndofs)
    # fixed = fixeddofs(resolution)
    # fixed = np.array(fixed)
    # free=np.setdiff1d(dofs,fixed)
    '''
    for i in range(ndofs):
        if K[idofidx,i] != 0:
            print(f" Uvec[{i}] * K[{idofidx},{i}] = {Uvec[i]} * {K[idofidx,i]}")
    '''
    # print(f"UKU = {np.ones((1,K.shape[0]),dtype = K.dtype)[:, free]@KU[free]}")
    
def FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz):
    
    
    print('-----------------------------------------FEconv_PyCheck_st')
    convresult = 0
    h8type = H8types[ibatch,0,Idxx,Idxy,Idxz]
    
    direction = outidx % 3
    print(f"h8type = {h8type}, direction = {direction}")
    for j in range(27):
        # uidx1 = nodIdx[Idxz][Idxy][Idxx][j][0];
        # uidx2 = nodIdx[Idxz][Idxy][Idxx][j][1];
        # uidx3 = nodIdx[Idxz][Idxy][Idxx][j][2];
        uidx1 = nodIdx[Idxx][Idxy][Idxz][j][0];
        uidx2 = nodIdx[Idxx][Idxy][Idxz][j][1];
        uidx3 = nodIdx[Idxx][Idxy][Idxz][j][2];
        if ((uidx1+1)*(uidx2+1)*(uidx3+1)!=0):
            # print(f' j={j}, uidx1={uidx1}, uidx2={uidx2}, uidx3={uidx3}')
            for ix in range(3):
                # print(f'ix={ix}, j={j}, uidx1={uidx1}, uidx2={uidx2}, uidx3={uidx3}')
                # convresult += U[ibatch][outidx - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][ix][direction][j];
                # convresult += U[ibatch][outidx - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][direction][ix][j];
                convresult += U[ibatch][outidx - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][direction][ix][j];
                # ix=0
                # print(f"U[{outidx - direction + ix}][{uidx1}][{uidx2}][{uidx3}] * filters[{direction}][{ix}][{j}] = {U[ibatch][outidx - direction + ix][uidx1][uidx2][uidx3]} * {filters[h8type][direction][ix][j]}")
    print('convresult = ',convresult.numpy())
    # tmp = filters[10,:,2,:]
    # print(tmp.sum(),tmp[:,[1,2,4,5,7,8,10,11,13,14,16,17]].sum())
    
    return convresult



#%% main
if __name__ == "__main__":

    print('modify mark 0')
    
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)

    ibatch = 0; outidx = 1; Idxx = 31; Idxy = 11; Idxz = 21;
    #%%% rho,U generate 
    # ========================================================================
    batchsize = 64; resolution = 40
    randU=True; randrho=True; isFloat64=True
    if isFloat64:
        torchtype = torch.float64
    else:
        torchtype = torch.float32
    print(f'**** batchsize = {batchsize}, resolution = {resolution} ****')
    if randU:
        U = torch.rand((batchsize,18,resolution,resolution,resolution),dtype = torchtype,device=device)
    else:
        U = torch.ones((batchsize,18,resolution,resolution,resolution),dtype = torchtype,device=device)
    if randrho:
        rho = torch.rand((batchsize,1,resolution,resolution,resolution),dtype = torchtype)
    else:
        rho = torch.ones((batchsize,1,resolution,resolution,resolution),dtype = torchtype)
    
    from periodicU import periodicU
    U = periodicU(U)
    U.requires_grad_()
    
    rho[rho<=0.5] = 0
    rho[rho >0.5] = 1
    print(f"U is random: **{randU}**, rho is random: **{randrho}**")
    print(f'rho is discret: **{torch.abs((rho-1)*rho).max().item() == 0}**' )
    print(f"U.requires_grad={U.requires_grad}, U.is_leaf = {U.is_leaf}")
    
    #%%% Hom. Param.s
    # ========================================================================
    Ke,Fe,edofMat = data_pre_dtype(batchsize = 1,resolution = resolution, isFloat64=True)

    # import h5py
    # matFile = "G:\FangCloudV2\个人文件\WorkFiles\MMC_DNN\morphology\Kematlab.mat"
    # matData = h5py.File(matFile,'r')
    # Ke = np.transpose( matData['Ke'][()])
    # Ke = np.random.rand(24,24)
    # for i in range(24):
    #     for j in range(i,24):
    #         Ke[i,j] = Ke[j,i]
    # print('* Ke	:',Ke.shape,Ke.dtype,abs(Ke).sum())
    
    # filtercheck(Ke.astype(np.float64))

    
    #%%% FEconv_test
    print("======================= FEconvNet ===============================")
    
    FEconv_runTwice(U,rho,Ke)
    
    # FEconv_Pycheck_varU(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    
    # feconv_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    L = feconvNet_periodicU_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    # FEconv_PyCheck(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    # ix = outidx % 3
    # idxx = list(np.arange(ix,24,3))
    # idxy = list(np.arange(24))
    # # idxy = list(np.arange(iy,24,3))
    # print(Ke[idxx,:][:,idxy].sum(),Ke[idxy,:][:,idxx].sum())
    

    print("------------------------ Gradients ------------------------------")
    start = time.perf_counter()

    print("L = ",L)
    print(f"U.requires_grad={U.requires_grad}, U.is_leaf = {U.is_leaf}")
    L.backward()
    gradU = U.grad
    print(f"gradU.shape = {gradU.shape}")
    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 11; Idxz = 21;
    print(f"*** gradU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = {gradU[ibatch,outidx,Idxx,Idxy,Idxz]}")
    print(f"*** gradU.sum() = {gradU.sum()}")
    elapsed = time.perf_counter() - start
    print(f"elapsed in {elapsed}s")
    
    # FEconv_PyCheck(U,rho,Ke)
    
    FEconv_runTwice(U,rho,Ke)
    
'''
    #%%% original_test
    print('==================== original method ============================')
    L = oricheck(U,rho,Ke,edofMat)
    print("------------------------ Gradients ------------------------------")
    start = time.perf_counter()
    print("L = ",L)
    print(f"U.requires_grad={U.requires_grad}, U.is_leaf = {U.is_leaf}")
    L.backward()
    gradU = U.grad
    print(f"gradU.shape = {gradU.shape}")
    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 11; Idxz = 21;
    print(f"*** gradU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = {gradU[ibatch,outidx,Idxx,Idxy,Idxz]}")
    print(f"*** gradU.sum() = {gradU.sum()}")
    elapsed = time.perf_counter() - start
    print(f"elapsed in {elapsed}s")
    
    #%%% assemble_test
    print('===================== assembleKU ================================')
    assembleKU_check(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz)

    assembleKU_periodic(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz)
'''