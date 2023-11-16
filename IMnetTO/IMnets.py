# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:36:00 2021

@author: Liangchao Zhu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)
def module_size(module):
    assert isinstance(module, torch.nn.Module)
    n_params, n_conv_layers = 0, 0
    for name, param in module.named_parameters():
        if 'conv' in name:
            n_conv_layers += 1
        n_params += param.numel()
    return n_params, n_conv_layers

class G_CNN3d(nn.Module):
    def __init__(self, input_dims=256, dense_channels=32, nelx=20,nely=20,nelz=20):
        super(G_CNN3d, self).__init__()
        self.features = nn.Sequential()

        # resizes = (1, 2, 1, 2, 1, 2, 1)
        # conv_filters = (128, 64, 32, 32, 16, 16, 1)
        resizes = (1, 2, 1, 2, 1)
        conv_filters = (128, 64, 32, 16, 1)
        # 多加几层conv
        # conv_filters = (128, 64, 1)
        # 5, 10, 20, 40
        fx = nelx//4
        fy = nely//4
        fz = nelz//4
        dense_filters = fx*fy*fz * dense_channels

        self.features.add_module('In_dense', nn.Linear(input_dims, dense_filters))
        self.features.add_module('reshape', View((-1, dense_channels, fx, fy, fz))) # bs,32,5,5,5

        pre_filter = dense_channels
        i = 1
        # for filters in conv_filters:
        for resize in resizes:
            filters = conv_filters[i-1]
            self.features.add_module('Block%d_Act' % i, nn.Tanh())
            if resize == 2:
                self.features.add_module('Block%d_Up' % i, nn.Upsample(scale_factor=2, mode='trilinear'))
            self.features.add_module('Block%d_BN' % i, nn.BatchNorm3d(pre_filter))
            self.features.add_module('Block%d_conv' % i, nn.Conv3d(pre_filter, filters, kernel_size=5, stride=1, padding=2,
                                         bias=True, padding_mode='replicate'))
            # bias?
            pre_filter = filters
            i += 1

        # self.input = nn.Linear(input_dims, dense_filters)
        # self.Trans = View((-1, dense_channels, 5, 5, 5))

        print('# params {}, # conv layers {}'.format(
            *self.model_size))

    def forward(self, x):
        return F.sigmoid(self.features(x))

    @property
    def model_size(self):
        return module_size(self)


class generator_NAND_rand(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_NAND_rand, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)

	def forward(self, points):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)
		# print("--------- l7: ",l7.min().item(),l7.max().item())

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l7 = torch.sigmoid(l7)
		
		# l7 = (torch.tanh(l7)+1)/2
		return l7


class generator_NAND(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_NAND, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)

	def forward(self, points):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)
		print("--------- l7: ",l7.min().item(),l7.max().item())

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		# l7 = torch.sigmoid(l7)
		
		# l7 = (torch.tanh(l7)+1)/2
		return l7

class generator_NAND_U(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_NAND_U, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 3, bias=True)
		# nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_1.bias,0)
		# nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_2.bias,0)
		# nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_3.bias,0)
		# nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_4.bias,0)
		# nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_5.bias,0)
		# nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		# nn.init.constant_(self.linear_6.bias,0)
		# nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		# nn.init.constant_(self.linear_7.bias,0)

	def forward(self, points):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		# l7 = torch.sigmoid(l7)
		
		return l7


class generator_SAND(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_SAND, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 4, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)

	def forward(self, points):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l7[:,0] = torch.sigmoid(l7[:,0])
		
		return l7



class generator_SAND_rand(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_SAND_rand, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 4, bias=True)

	def forward(self, points):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, inplace=True)

		l7 = self.linear_7(l6)

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l7[:,0] = torch.sigmoid(l7[:,0])
		
		return l7

class generator_SAND_seperate(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_SAND_seperate, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)

		self.linear_1b = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2b = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3b = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4b = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5b = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6b = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7b = nn.Linear(self.gf_dim*1, 3, bias=True)


	def forward(self, pointz):
		# zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# pointz = torch.cat([points,zs],2)
		# pointz = points
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)

		#l7 = torch.clamp(l7, min=0, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l7[:,0] = torch.sigmoid(l7[:,0])
		

		l1b = self.linear_1b(pointz)
		l1b = F.leaky_relu(l1b, inplace=True)

		l2b = self.linear_2b(l1b)
		l2b = F.leaky_relu(l2b, inplace=True)

		l3b = self.linear_3b(l2b)
		l3b = F.leaky_relu(l3b, inplace=True)

		l4b = self.linear_4b(l3b)
		l4b = F.leaky_relu(l4b, inplace=True)

		l5b = self.linear_5b(l4b)
		l5b = F.leaky_relu(l5b, inplace=True)

		l6b = self.linear_6b(l5b)
		l6b = F.leaky_relu(l6b, inplace=True)

		l7b = self.linear_7b(l6b)

		return torch.cat((l7,l7b),1)