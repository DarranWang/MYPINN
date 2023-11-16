# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 11:25:23 2021

@author: Liangchao Zhu
"""
import numpy as np
from torch import nn
from torch.autograd import Function
import torch

import feconv_cuda
import feconvR_cuda

from periodicU import periodicU


from getTypeH8 import typeH8_
from getTypeH8 import kernelForType
from arrangeIndex import arrangeIndex
from symbolicExec_vec2 import getFilters,getFiltersFE


''' 
#forward#
	INPUT:
		* U 		: [batch_size,18,41,41,41]
		* H8types	: [batch_size, 1,41,41,41]
		* nodIdx	: [41,41,41,27,3]
		* filters	: [2^8, 3*3,27]

	OUTPUT:
		* KU 		: [batch_size,18,41,41,41]
'''


class FEconvFunction(Function):
	@staticmethod
	def forward(ctx,U,H8types,nodIdx,filters):
        
		outputs = feconv_cuda.forward(U,H8types,nodIdx,filters)
		V = outputs[0]
		ctx.save_for_backward(*outputs)
		return V
	@staticmethod
	def backward(ctx,gradV):
		outputs = feconv_cuda.backward(gradV,*ctx.saved_variables)
		return outputs[0]
# https://blog.csdn.net/littlehaes/article/details/103828130
# ctx.save_for_backward(a, b)能够保存forward()静态方法中的张量, 
# 从而可以在backward()静态方法中调用, 
# 具体地, 下面地代码通过a, b = ctx.saved_tensors重新得到a和b


class FECONV(nn.Module):
	def __init__(self):
		super(FECONV,self).__init__()
	def forward(self,U,H8types,nodIdx,filters):
		return FEconvFunction.apply(U,H8types,nodIdx,filters)
    
    
class FEconvLayer(Function):
    @staticmethod
    def forward(ctx,U,rho,nodIdx,filters,typeFilter):
        U = periodicU(U)
        H8types = typeH8_(rho,typeFilter)
        # H8types = H8types.int()
        outputs = feconv_cuda.forward(U,H8types,nodIdx,filters)
        KU = outputs[0]
        ctx.save_for_backward(*outputs)
        return KU,U
    @staticmethod
    def backward(ctx,gradU):
        outputs = feconv_cuda.backward(gradU,*ctx.saved_variables)
        return outputs[0]
class FEconvNet(nn.Module):
    def __init__(self,datatype=np.float64,device=torch.device("cuda:0")):
        super(FEconvNet,self).__init__()
        self.typeFilter = torch.from_numpy(kernelForType(datatype)[np.newaxis,np.newaxis]).to(device)
    def forward(self,U,rho,nodIdx,filters):
        return FEconvLayer.apply(U,rho,nodIdx,filters,self.typeFilter)


class FEconvLayer_periodicU(Function):
    @staticmethod
    def forward(ctx,U,rho,nodIdx,filters,typeFilter):
        H8types = typeH8_(rho,typeFilter)
        # H8types = H8types.int()
        outputs = feconv_cuda.forward(U,H8types,nodIdx,filters)
        V = outputs[0]
        variables = [H8types,nodIdx,filters]
        ctx.save_for_backward(*variables)
        return V
    @staticmethod
    def backward(ctx,gradV):
        outputs = feconv_cuda.backward(gradV,*ctx.saved_variables)
        return outputs[0],None,None,None,None
class FEconvNet_periodicU(nn.Module):
    def __init__(self,datatype=np.float64,device=torch.device("cuda:0")):
        super(FEconvNet_periodicU,self).__init__()
        self.typeFilter = torch.from_numpy(kernelForType(datatype)[np.newaxis,np.newaxis]).to(device)
    def forward(self,U,rho,nodIdx,filters):
        return FEconvLayer_periodicU.apply(U,rho,nodIdx,filters,self.typeFilter)


class FEconvLayer_periodicU_H8types(Function):
    @staticmethod
    def forward(ctx,U,H8types,nodIdx,filters,typeFilter):
        # H8types = typeH8_(rho,typeFilter)
        # H8types = H8types.int()
        outputs = feconv_cuda.forward(U,H8types,nodIdx,filters)
        V = outputs[0]
        variables = [H8types,nodIdx,filters]
        ctx.save_for_backward(*variables)
        return V
    @staticmethod
    def backward(ctx,gradV):
        outputs = feconv_cuda.backward(gradV,*ctx.saved_variables)
        return outputs[0],None,None,None,None
class FEconvNet_periodicU_H8types(nn.Module):
    def __init__(self,Ke,datatype=np.float64,device=torch.device("cuda:0")):
        super(FEconvNet_periodicU_H8types,self).__init__()
        self.typeFilter = torch.from_numpy(kernelForType(datatype)[np.newaxis,np.newaxis]).to(device)
        self.nodIdx = torch.from_numpy(arrangeIndex()).to(device)
        self.filters = torch.from_numpy(getFilters(Ke)).to(device)
    def forward(self,U,H8types):
        return FEconvLayer_periodicU_H8types.apply(U,H8types,self.nodIdx,self.filters,self.typeFilter)


# https://blog.csdn.net/tsq292978891/article/details/79364140
class FEconvLayerFE(Function):
	@staticmethod
	def forward(ctx,U,H8types,FEfilters):
		outputs = feconvR_cuda.forward(U,H8types,FEfilters)
		variables = [H8types,FEfilters]
		ctx.save_for_backward(*variables)
		return outputs[0]
	@staticmethod
	def backward(ctx,gradVfe):
		gradVfe = gradVfe.contiguous()
		outputs = feconvR_cuda.backward(gradVfe,*ctx.saved_variables)
		return outputs[0],None,None
class FEconvModuleFE(nn.Module):
	def __init__(self,FE,device=torch.device("cuda:0")):
		super(FEconvModuleFE,self).__init__()
		self.FEfilters = torch.from_numpy(getFiltersFE(FE)).to(device)
	def forward(self,U,H8types):
		return FEconvLayerFE.apply(U,H8types,self.FEfilters)
