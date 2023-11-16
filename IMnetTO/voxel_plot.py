# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:58:58 2021

@author: Liangchao Zhu
"""
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
def voxel_plot(Voxel):
    ax = plt.figure().add_subplot(111, projection = '3d')
    ax.voxels(Voxel)
    plt.show()
def voxel_savefig(Voxel,filepath):
    ax = plt.figure().add_subplot(111, projection = '3d')
    ax.voxels(Voxel)
    plt.savefig(filepath)

def DispField_plot(u,filepath):
    fig, axes = plt.subplots(2,3)
    titles = ["X","Y","Z"]
    for i in range(3):
        ax=axes[0,i].imshow(u[i,:,:,0],vmin=u[i].min(), vmax=u[i].max())
        axes[0,i].set_title(titles[i])
        ax=axes[1,i].imshow(u[i,:,:,-1],vmin=u[i].min(), vmax=u[i].max())
        fig.colorbar(ax,ax=(axes[0,i],axes[1,i]),orientation='horizontal')
    # plt.show()
    plt.savefig(filepath)
    plt.close(fig)
    
def StrDisp_plot(x,u,filepath):
    fig, axes = plt.subplots(3,3)
    axes[0,0].cla()
    axes[0,1].cla()
    axes[0,2].cla()
    ax=axes[0,0].imshow(x[0,0,0,:,:],vmin=0, vmax=1)
    ax=axes[0,1].imshow(x[0,0,:,0,:],vmin=0, vmax=1)
    ax=axes[0,2].imshow(x[0,0,:,:,0],vmin=0, vmax=1)
    
    fig.colorbar(ax,ax=(axes[0,0],axes[0,1],axes[0,2]))
    
    titles = ["X","Y","Z"]
    for i in range(3):
        ax=axes[1,i].imshow(u[i,:,:,0],vmin=u[i].min(), vmax=u[i].max())
        axes[1,i].set_title(titles[i])
        ax=axes[2,i].imshow(u[i,:,:,-1],vmin=u[i].min(), vmax=u[i].max())
        fig.colorbar(ax,ax=(axes[1,i],axes[2,i]),orientation='horizontal')
    # plt.show()
    plt.savefig(filepath)
    plt.close(fig)
    
def Grads_plot(x,filepath):
    fig, axes = plt.subplots(1,3)
    axes[0].cla()
    axes[1].cla()
   
    ax=axes[0].imshow( x[-1,:,:])
    ax=axes[1].imshow( x[:,-1,:])
    ax=axes[2].imshow( x[:,:,-1])
    
    fig.colorbar(ax,ax=axes[0],orientation='horizontal')
    fig.colorbar(ax,ax=axes[1],orientation='horizontal')
    fig.colorbar(ax,ax=axes[2],orientation='horizontal')
    plt.savefig(filepath)
    plt.close(fig)