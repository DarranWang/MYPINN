# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 11:25:23 2021

@author: Liangchao Zhu
"""
from torch import nn
from torch.autograd import Function
import torch

import feconv_cuda

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
		KU = outputs[0]
		ctx.save_for_backward(*outputs)
		return KU
	@staticmethod
	def backward(ctx,gradU):
		outputs = feconv_cuda.backward(gradU,*ctx.saved_variables)
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