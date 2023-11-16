from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import time
#import feconv_cuda


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


def data_pre_dtype(batchsize = 1,resolution = 40,isFloat64=False):
    U,rho,Ke,Fe,edofMat = data_pre(batchsize,resolution)
    if isFloat64:
        U = U.double()
        rho = rho.double()
        Ke = Ke.astype(np.float64)
        Fe = Fe.astype(np.float64)
    return U,rho,Ke,Fe,edofMat
def data_pre(batchsize = 1,resolution = 40):
    # resolution = 40
    print(f'**** batchsize = {batchsize}, resolution = {resolution} ****')
    U = torch.rand((batchsize,18,resolution,resolution,resolution),dtype = torch.float)
    # U = torch.ones((batchsize,18,resolution,resolution,resolution),dtype = torch.float)
    rho = torch.rand((batchsize,1,40,40,40),dtype = torch.float)
    # rho = torch.ones((batchsize,1,resolution,resolution,resolution),dtype = torch.float)
    rho[rho<=0.5] = 0
    rho[rho >0.5] = 1
    print('discret rho:', torch.abs((rho-1)*rho).max())
    
    # resolution = 40 
    E = 1e6; nu = 0.3
    from ConstMatricesForHomogenization import ISOelasticitytensor,LocalKeFe
    D0 = ISOelasticitytensor(E, nu)
    Ke,Fe,intB = LocalKeFe(resolution,D0)
    h = 1.0/resolution
    nele = resolution**3
    I = np.eye(6)
    datashape = resolution
    # Ke = torch.from_numpy(Ke).to(device)
    # Fe = torch.from_numpy(Fe).to(device)
    
    from PeriodicMesh3D import PeriodicMesh3D,edofMatrix
    eleidx,MESH,V = PeriodicMesh3D(resolution)
    # from mesh3D import mesh3D,edofMatrix
    # eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    
    print('====================== data prepared for homogenization')
    # Ke = np.ones(Ke.shape,dtype = Ke.dtype)
    # Ke = np.eye(24,dtype = Ke.dtype)
    # Ke[0,1] = 1
    return U,rho,Ke,Fe,edofMat

def originalMethod_check(output_img,input,Ke, edofMat):
    
    size = input.shape[0]
    # 3d rho的顺序?
    pp = input.view(size, -1, 1, 1)#.to(device)
    K = pp * Ke  # [bs, 8000, 24, 24]
    # F = pp * Fe # [bs, 8000, 24, 6]
    
    ref18 = output_img.contiguous().view(size,18,-1) #[bs,18,40**3]
    
    map0 = ref18[:,0:3].permute((0,2,1)).contiguous().view(size,-1,1)
    map1 = ref18[:,3:6].permute((0,2,1)).contiguous().view(size,-1,1)
    map2 = ref18[:,6:9].permute((0,2,1)).contiguous().view(size,-1,1)
    map3 = ref18[:,9:12].permute((0,2,1)).contiguous().view(size,-1,1)
    map4 = ref18[:,12:15].permute((0,2,1)).contiguous().view(size,-1,1)
    map5 = ref18[:,15:18].permute((0,2,1)).contiguous().view(size,-1,1)
    
    ref_map = torch.cat([map0,map1,map2,map3,map4,map5], 2)# [bs,3*40**3,6]
    # ref_map[:,0:3,:] = 0

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
    print(losst1)
    return losst1

def datapre_feconv(U,rho,Ke):
    from periodicU import periodicU
    U = periodicU(U)
    
    from getTypeH8 import typeH8
    H8types = typeH8(rho)
    H8types = H8types.int()

    from arrangeIndex import arrangeIndex
    nodIdx = arrangeIndex()
    nodIdx = nodIdx.astype(np.int32)

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
    print(uku1.cpu().numpy()[0,0])
    sum0 = uku[0,:3].sum()
    print(uku1.shape, uku1[0,0]-sum0)
    print(uku1[:2])
    
    # ibatch = 0; outidx = 1; Idxx = 1; Idxy = 2; Idxz =3;
    FEconv_PyCheck_st(U,H8types,filters,nodIdx,ibatch,outidx,Idxx,Idxy,Idxz)
    print(f'KU[{ibatch},{outidx},{Idxx},{Idxy},{Idxz}] = ',KU[ibatch,outidx,Idxx,Idxy,Idxz])
    
def oricheck(U,rho,Ke,edofMat):
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print("DEVICE : ", device)
    edofMat = torch.from_numpy(edofMat).to(device).long()
    Ke = torch.from_numpy(Ke).to(device)
    rho = rho.to(device)
    U = U.to(device)
    
    print("INPUT info.:----------------------------------")
    print('* U	    :',U.cpu().numpy().shape,U.dtype,U.sum().cpu().numpy())
    print('* rho	:',rho.cpu().numpy().shape,rho.dtype,rho.sum().cpu().numpy(),rho.min().cpu().numpy(),rho.max().cpu().numpy())
    print('* edofMat	:',edofMat.cpu().numpy().shape,edofMat.dtype,edofMat.sum().cpu().numpy(),edofMat.min().cpu().numpy(),edofMat.max().cpu().numpy())
    print('* Ke	    :',Ke.cpu().numpy().shape,Ke.dtype,abs(Ke).sum().cpu().numpy())
    
    start = time.perf_counter()
    losst1 = originalMethod_check(U,rho,Ke, edofMat)
    elapsed = time.perf_counter() - start
    print(f"elapsed in {elapsed} s")

def assembleKU_check(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz):
    x = rho[0,0].cpu().numpy()
    resolution = x.shape[0]
    nele = resolution**3; ndof = nele#3*(resolution+1)**3
    K,F = AssembleGlobalKF(x,Ke,Fe,edofMat,ndof,nele)
    print(K.dtype,K.shape)
    
    
    ref_map = Utensor2vec(U)
    
    uku = np.zeros((6))
    for i in range(6):
        Uvec = ref_map[0,:,[i]].cpu().numpy()
        uku[i] = Uvec.T @ (K@Uvec)
    print(uku)
    
    imap = outidx//3
    Uvec = ref_map[0,:,[imap]].cpu().numpy()
    KU = K@Uvec
    inodidx = Idxx*resolution**2 + Idxy*resolution + Idxz
    idofidx = inodidx*3 + outidx % 3
    print('*** assembleKU_i = ',KU[idofidx])
    
    # Uvec = np.ones((K.shape[0],1))
    # print(Uvec[3:].T @ (K[3:,:][:,3:]@Uvec[3:]))
    # Uvec[:3] = 0
    # print(Uvec.T @ (K@Uvec))

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

if __name__ == "__main__":
    # KUperEleComputation()
    
    
    print('modify mark 1')
    resolution = 40
    U,rho,Ke,Fe,edofMat = data_pre_dtype(batchsize = 1,resolution = resolution, isFloat64=True)
    
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

    ibatch = 0; outidx = 2; Idxx = 40; Idxy = 1; Idxz = 0;
    
    
    # FEconv_Pycheck_varU(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    
    # feconv_check(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    FEconv_PyCheck(U,rho,Ke,ibatch,outidx,Idxx,Idxy,Idxz)
    # ix = outidx % 3
    # idxx = list(np.arange(ix,24,3))
    # idxy = list(np.arange(24))
    # # idxy = list(np.arange(iy,24,3))
    # print(Ke[idxx,:][:,idxy].sum(),Ke[idxy,:][:,idxx].sum())
    print('==========================================================')
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # from periodicU import periodicU
    # U = periodicU(U)
    # U = U.to(device)
    # tmp = U*U
    # print('tmp: ',tmp.shape,tmp.device)
    
    # print((U*U).sum())
    # oricheck(U,rho,Ke,edofMat)
    
    
    # assembleKU_check(rho,U,Ke,Fe,edofMat,ibatch,outidx,Idxx,Idxy,Idxz)
    # FEconv_PyCheck(U,rho,Ke)
    '''    '''
