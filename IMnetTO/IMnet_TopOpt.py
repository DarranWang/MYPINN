# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:58:27 2021

@author: Liangchao Zhu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from IMnets import generator_NAND_U,generator_NAND_rand,G_CNN3d
from voxel_plot import voxel_savefig,DispField_plot,StrDisp_plot,Grads_plot

from practices import OneCycleScheduler, adjust_learning_rate

import time

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# plt.ion()

nelxyz = 20

def analyticalGrads(x,u,edofMat,nelx,nely,nelz,KE):
    import numpy as np
    u = Utensor2vec(u).view(-1,1).detach().cpu().numpy()# torch.Size([1, 27783, 1])
    ce = (np.dot(u[edofMat].reshape(nelx * nely * nelz, 24), KE) * u[edofMat].reshape(nelx * nely * nelz, 24)).sum(1)
    # print(ce.sum())
    penal = 1;Emax = 1; Emin = 1e-3
    dc = (-penal * x.flatten() ** (penal - 1) * (Emax - Emin)) * ce
    # print(dc.shape)
    return dc.reshape(nelx,nely,nelz)

def numericalSol(x,edofMat,nelx,nely,nelz,KE,f):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    penal = 1;Emax = 1; Emin = 1e-3
    
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
    u = np.zeros((ndof, 1),dtype=float)
    
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()
    sK = ((KE.flatten()[np.newaxis]).T * (Emin + (x) ** penal * (Emax - Emin))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    
    
    K = K[free, :][:, free]
    u[free, 0] = spsolve(K, f[free, 0])
    
def PointsSampling(nelx,nely,nelz):
    x,y,z = torch.meshgrid(torch.arange(nelx+1),
                           torch.arange(nely+1),
                           torch.arange(nelz+1))
    points = torch.cat((x.contiguous().view(-1,1),
               y.contiguous().view(-1,1),
               z.contiguous().view(-1,1)),1)
    return points.float()

class ConvNode2Elem(nn.Module):
    def __init__(self,device=torch.device("cpu")):
        super(ConvNode2Elem,self).__init__()
        self.filterKernel = torch.ones((1,1,2,2,2),dtype=torch.float,device=device)*0.125
    def forward(self,x):
        return F.conv3d(x, self.filterKernel,padding=0)

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

def constParameters():
    resolution = 1 
    E = 1; nu = 0.3
    from elasticity3d import ISOelasticitytensor,LocalKeFe
    D0 = ISOelasticitytensor(E, nu)
    Ke,Fe,intB = LocalKeFe(resolution,D0)
    
    resolution = nelxyz
    from mesh3D import mesh3D,edofMatrix
    eleidx,MESH,V = mesh3D(resolution)
    edofMat = edofMatrix(MESH)
    
    return Ke,Fe,edofMat


def element_wise_uku(x,u,Ke,edofMat):
    # x = x.permute((0,1,4,3,2))
    bs = x.shape[0]
    x = x.contiguous().view(bs,-1,1,1)
    K = (x ** 1) * Ke   # [bs, 8000, 24, 24]
    u = Utensor2vec(u)
    # print(u.shape)
    u = u[:, edofMat, :] # torch.Size([1, 64000, 24, 1])
    # print(u.shape)
    uT = u.permute([0, 1, 3, 2])
    
    UKU = torch.matmul(torch.matmul(uT, K), u).sum()
    return UKU

def DirichletBC(u):
    u[:,:,0,:,:] = 0.0
    return u

def NeumannBC(nelx,nely,nelz):
    f = torch.zeros((1,3,nelx+1,nely+1,nelz+1))
    f[:,1,nelx,nely,:] = -1
    return f

def plottt(gs):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(len(gs)),gs)
    # plt.show()
    plt.ylim(0,np.max(gs)*1.2)
    plt.savefig('./figs/gs')

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def trainSAND_seperate_CNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # torch.manual_seed(1)
    nelx=nelxyz;nely=nelxyz;nelz=nelxyz;
    
    z_dim = 0; point_dim = 3; gf_dim = 16;
    # model_x = generator_NAND(z_dim, point_dim, gf_dim).to(device)
    
    var_x = torch.rand([1, 256]).to(device)
    model_x = G_CNN3d(input_dims=256, dense_channels=24, nelx=nelx,nely=nely,nelz=nelz).to(device)
    
    model_U = generator_NAND_U(z_dim, point_dim, gf_dim).to(device)
    
    # print(IMgenerator)
    
    # coors = torch.randn(128,3)*100
    
    points = PointsSampling(nelx,nely,nelz).to(device)
    structMap = ConvNode2Elem(device).to(device)

    Ke, _, edofMat = constParameters()
    Ke = torch.from_numpy(Ke).to(device)
    
    f = NeumannBC(nelx,nely,nelz).to(device)
    
    w_equilibrium = 5
    w_vol = 1e1
    volfrac = 0.6
    
    optimizer_U = torch.optim.SGD(model_U.parameters(), lr=1e-5, momentum=0.9)
    optimizer_x = torch.optim.SGD(model_x.parameters(), lr=1e-1, momentum=0.9)
    
    epochs = 3000
    xstep=100
    tt=0
    gs = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        isXstep = epoch % xstep >= xstep-1
        

        model_x.train()
        model_x.zero_grad()
        # y_x = model_x(points).permute((1,0)).view(1,1,nelx+1,nely+1,nelz+1)
        # x = structMap(y_x)
        # x.register_hook(save_grad('x'))
        
        x = model_x(var_x)
        x.register_hook(save_grad('x'))
        
        model_U.train()
        model_U.zero_grad()
        y_U = model_U(points).permute((1,0)).view(1,3,nelx+1,nely+1,nelz+1)
        u = DirichletBC(y_U)
        FU = (f*u).sum()
        
        # y = torch.cat(model_x(points),model_U(points),1)
        # y=y.permute((1,0)).view(1,4,nelx+1,nely+1,nelz+1)

        UKU = element_wise_uku(x,u,Ke,edofMat)
        
        # # SANDNL
        # loss = UKU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
        # print(f"epoch {epoch} - loss: {loss:e}, [1]: {UKU:e}, [2]: {w_equilibrium*(UKU/2-FU):e}, [3]: {w_vol*(x.mean()-volfrac)**2:e}")
        # print(f"---------- loss: {loss:e}, comp.: {UKU:e}, eng.: {UKU/2-FU:e}, vol.: {x.mean():.3f}")
        # print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}")
        
        # SAND
        
        if isXstep:
            # loss = FU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
            # loss = FU + w_vol*((x.mean()-volfrac)**2)
            # loss = w_vol*((x.mean()-volfrac)**2)
            w_vol *= 1.5
            loss = - UKU + w_vol*((x.mean()-volfrac)**2)
        else:
            loss = FU#UKU/2#-FU
        if isXstep:
            print(f"epoch [{epoch}] - loss: {loss:.3e}, [1]: {FU:.3e}, [2]: {w_equilibrium*(UKU/2-FU):.3e}, [3]: {w_vol*(x.mean()-volfrac)**2:.3e}")
            print(f"---------- equ.={UKU/FU:.3f}, eng.: {UKU/2-FU:.3e}, vol.: {x.mean():.3f}, w_vol: {w_vol}")
            print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}, [{x.max().item()-x.min().item():.3f}]")
        
        gs.append(x.max().item()-x.min().item())
        
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        
        if isXstep:
            gradx = grads['x']
            #print(gradx.shape)#torch.Size([1, 1, 21, 21, 21])
            optimizer_x.step()
            # voxel_savefig(x.squeeze()>=x.mean(),'./figs/struct'+str(epoch))
            # DispField_plot(u[0].cpu().detach().numpy(),'./figs/DispSlice'+str(epoch))
            StrDisp_plot(x.cpu().detach().numpy(),u[0].cpu().detach().numpy(),'./figs/DispSlice'+str(epoch))
            
            dc = analyticalGrads(x.cpu().detach().numpy(),u,edofMat,nelx,nely,nelz,Ke.cpu().numpy())
            Grads_plot(dc,'./figs/GradAnay'+str(epoch))
            Grads_plot(gradx[0,0].cpu().detach().numpy(),'./figs/Grad'+str(epoch))
            # Grads_plot(1+2*gradx[0,0].cpu().detach().numpy()/dc,'./figs/GradSum'+str(epoch))
            Grads_plot(1-gradx[0,0].cpu().detach().numpy()/dc,'./figs/GradSum'+str(epoch))
            print(UKU.item(),FU.item())
            
        else:
            optimizer_U.step()
        torch.cuda.synchronize()
        tt += time.perf_counter() - start
        
        # if epoch % 1000 == 1:
        #     voxel_savefig(x.squeeze()>=x.mean(),'./figs/struct'+str(epoch))
        
    print(f'elapsed in {tt}s')
    # plottt(gs)


def trainSAND_seperate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # torch.manual_seed(1)
    
    z_dim = 0; point_dim = 3; gf_dim = 16;
    model_x = generator_NAND_rand(z_dim, point_dim, gf_dim).to(device)
    model_U = generator_NAND_U(z_dim, point_dim, gf_dim).to(device)
    
    # print(IMgenerator)
    
    # coors = torch.randn(128,3)*100
    nelx=nelxyz;nely=nelxyz;nelz=nelxyz;
    points = PointsSampling(nelx,nely,nelz).to(device)
    structMap = ConvNode2Elem(device).to(device)

    Ke, _, edofMat = constParameters()
    Ke = torch.from_numpy(Ke).to(device)
    
    f = NeumannBC(nelx,nely,nelz).to(device)
    
    w_equilibrium = 5
    w_vol = 1e1
    volfrac = 0.6
    
    optimizer_U = torch.optim.SGD(model_U.parameters(), lr=1e-5, momentum=0.9)
    optimizer_x = torch.optim.SGD(model_x.parameters(), lr=1e-4, momentum=0.9)
    
    epochs = 3000
    xstep=100
    tt=0
    gs = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        isXstep = epoch % xstep >= xstep-2
        

        model_x.train()
        model_x.zero_grad()
        y_x = model_x(points).permute((1,0)).view(1,1,nelx+1,nely+1,nelz+1)
        x = structMap(y_x)
        x.register_hook(save_grad('x'))

        model_U.train()
        model_U.zero_grad()
        y_U = model_U(points).permute((1,0)).view(1,3,nelx+1,nely+1,nelz+1)
        u = DirichletBC(y_U)
        FU = (f*u).sum()
        
        # y = torch.cat(model_x(points),model_U(points),1)
        # y=y.permute((1,0)).view(1,4,nelx+1,nely+1,nelz+1)

        UKU = element_wise_uku(x,u,Ke,edofMat)
        
        # # SANDNL
        # loss = UKU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
        # print(f"epoch {epoch} - loss: {loss:e}, [1]: {UKU:e}, [2]: {w_equilibrium*(UKU/2-FU):e}, [3]: {w_vol*(x.mean()-volfrac)**2:e}")
        # print(f"---------- loss: {loss:e}, comp.: {UKU:e}, eng.: {UKU/2-FU:e}, vol.: {x.mean():.3f}")
        # print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}")
        
        # SAND
        
        if isXstep:
            # loss = FU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
            # loss = FU + w_vol*((x.mean()-volfrac)**2)
            # loss = w_vol*((x.mean()-volfrac)**2)
            
            loss = - UKU + w_vol*((x.mean()-volfrac)**2)
            w_vol *= 1.5
        else:
            loss = UKU/2-FU
        if isXstep:
            print(f"epoch [{epoch}] - loss: {loss:.3e}, [1]: {- UKU:.3e}, [2]: {w_equilibrium*(UKU/2-FU):.3e}, [3]: {w_vol*(x.mean()-volfrac)**2:.3e}")
            print(f"---------- equ.={UKU/FU:.3f}, eng.: {UKU/2-FU:.3e}, vol.: {x.mean():.3f}, w_vol: {w_vol:.2e}")
            print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}, [{x.max().item()-x.min().item():.3f}]")
        
        gs.append(x.max().item()-x.min().item())
        
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        
        if isXstep:
            gradx = grads['x']
            #print(gradx.shape)#torch.Size([1, 1, 21, 21, 21])
            optimizer_x.step()
            voxel_savefig(x.squeeze()>=x.mean(),'./figs/struct'+str(epoch))
            # DispField_plot(u[0].cpu().detach().numpy(),'./figs/DispSlice'+str(epoch))
            StrDisp_plot(x.cpu().detach().numpy(),u[0].cpu().detach().numpy(),'./figs/DispSlice'+str(epoch))
            
            dc = analyticalGrads(x.cpu().detach().numpy(),u,edofMat,nelx,nely,nelz,Ke.cpu().numpy())
            Grads_plot(dc,'./figs/GradAnay'+str(epoch))
            Grads_plot(gradx[0,0].cpu().detach().numpy(),'./figs/Grad'+str(epoch))
            # Grads_plot(1+2*gradx[0,0].cpu().detach().numpy()/dc,'./figs/GradSum'+str(epoch))
            Grads_plot(1-gradx[0,0].cpu().detach().numpy()/dc,'./figs/GradSum'+str(epoch))
            print(UKU.item(),FU.item())
            
        else:
            optimizer_U.step()
        torch.cuda.synchronize()
        tt += time.perf_counter() - start
        
        # if epoch % 1000 == 1:
        #     voxel_savefig(x.squeeze()>=x.mean(),'./figs/struct'+str(epoch))
        
    print(f'elapsed in {tt}s')
    # plottt(gs)
    
def trainSAND():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # torch.manual_seed(1)
    
    z_dim = 0; point_dim = 3; gf_dim = 16;
    model = generator_SAND_seperate(z_dim, point_dim, gf_dim).to(device)
    
    # print(IMgenerator)
    
    # coors = torch.randn(128,3)*100
    nelx=nelxyz;nely=nelxyz;nelz=nelxyz;
    points = PointsSampling(nelx,nely,nelz).to(device)
    structMap = ConvNode2Elem(device).to(device)

    Ke, _, edofMat = constParameters()
    Ke = torch.from_numpy(Ke).to(device)
    
    f = NeumannBC(nelx,nely,nelz).to(device)
    
    w_equilibrium = 1e0
    w_vol = 1e1
    volfrac = 0.5
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    epochs = 501
    tt=0
    gs = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        model.zero_grad()
        y = model(points)
        y=y.permute((1,0)).view(1,4,nelx+1,nely+1,nelz+1)

        x = structMap(y[:,[0]])
        u = DirichletBC(y[:,1:])

        UKU = element_wise_uku(x,u,Ke,edofMat)
        FU = (f*u).sum()
        # # SANDNL
        # loss = UKU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
        # print(f"epoch {epoch} - loss: {loss:e}, [1]: {UKU:e}, [2]: {w_equilibrium*(UKU/2-FU):e}, [3]: {w_vol*(x.mean()-volfrac)**2:e}")
        # print(f"---------- loss: {loss:e}, comp.: {UKU:e}, eng.: {UKU/2-FU:e}, vol.: {x.mean():.3f}")
        # print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}")
        # SAND
        loss = FU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
        print(f"epoch [{epoch}] - loss: {loss:e}, [1]: {FU:e}, [2]: {w_equilibrium*(UKU/2-FU):e}, [3]: {w_vol*(x.mean()-volfrac)**2:e}")
        print(f"---------- loss: {loss:e}, comp.: {FU:e}, eng.: {UKU/2-FU:e}, vol.: {x.mean():.3f}")
        print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}, [{x.max().item()-x.min().item():.3f}]")
        
        gs.append(x.max().item()-x.min().item())
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        optimizer.step()
        torch.cuda.synchronize()
        tt += time.perf_counter() - start
        
        if epoch % 1000 == 1:
            voxel_savefig(x.squeeze()>=x.mean(),'./figs/struct'+str(epoch))
        
    print(f'elapsed in {tt}s')
    plottt(gs)

def trainU():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # torch.manual_seed(1)
    
    z_dim = 0; point_dim = 3; gf_dim = 16;
    model = generator_NAND_U(z_dim, point_dim, gf_dim).to(device)
    
    # print(IMgenerator)
    
    # coors = torch.randn(128,3)*100
    nelx=nelxyz;nely=nelxyz;nelz=nelxyz;
    x = torch.rand((1,1,nelx,nely,nelz),device=device)
    x = torch.clamp(x,1e-3,1)
    points = PointsSampling(nelx,nely,nelz).to(device)
    # structMap = ConvNode2Elem(device).to(device)
    
    Ke, _, edofMat = constParameters() 
    Ke = torch.from_numpy(Ke).to(device)
    
    f = NeumannBC(nelx,nely,nelz).to(device)
    
    lr=1e-5
    lr_div = 2
    lr_pct = 0.3
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = OneCycleScheduler(lr_max=lr, div_factor=lr_div, pct_start=lr_pct)
    
    
    epochs = 3000
    tt=0
    LOG_loss = []
    elx = 1; ely = 2; elz = 3;
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        model.zero_grad()
        y = model(points) #torch.Size([68921, 3])
        # y=y.permute((1,0)).view(1,4,nelx+1,nely+1,nelz+1)
        # x = structMap(y[:,[0]])
        # u = DirichletBC(y[:,1:])
        # print(y[elx*41*41+ely*41+elz,:].detach())
        y=y.permute((1,0)).view(1,3,nelx+1,nely+1,nelz+1)
        # print(y[0,:,elx,ely,elz].detach())
        
        # x = structMap(y[:,[0]])
        u = DirichletBC(y)        

        UKU = element_wise_uku(x,u,Ke,edofMat)
        FU = (f*u).sum()
        # # SANDNL
        # loss = UKU + w_equilibrium*(UKU/2-FU) + w_vol*(x.mean()-volfrac)**2
        # print(f"epoch {epoch} - loss: {loss:e}, [1]: {UKU:e}, [2]: {w_equilibrium*(UKU/2-FU):e}, [3]: {w_vol*(x.mean()-volfrac)**2:e}")
        # print(f"---------- loss: {loss:e}, comp.: {UKU:e}, eng.: {UKU/2-FU:e}, vol.: {x.mean():.3f}")
        # print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}")
        # SAND
        loss = UKU/2-FU
        
        
        loss.backward()
        
        pct = (epoch) / (epochs)
        # print(step ,total_steps)
        lr = scheduler.step(pct)
        adjust_learning_rate(optimizer, lr)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e4)
        
        print(f"epoch [{epoch}] - loss: {loss:.3e}, UKU={UKU:.3e}, FU={FU:.3e}, equ.={UKU/FU:.3f}")
        # print(f"---------- {x.min().item():.3f} <= x <= {x.max().item():.3f}")
        print(f"---------- lr = {lr:.3e}")
        
        optimizer.step()
        torch.cuda.synchronize()
        tt += time.perf_counter() - start
        LOG_loss.append(loss.item())
    print(f'elapsed in {tt}s')

if __name__ == "__main__":
    # trainU()
    trainSAND_seperate()