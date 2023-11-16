# -*- coding: utf-8 -*-
"""
Created on Fri Feb 5 21:00:23 2021

@author: Liangchao Zhu
"""

import numpy as np

def getFiltersFE(FE):
    filter_list = []
    for i in range(6):
        filters = getFiltersFe(FE[:,[i]])
        filter_list.append(filters)
    Filters = np.concatenate(filter_list,axis=1)
    return Filters

def getFiltersFe(Fe):
    Filters = np.zeros((256,3),dtype = Fe.dtype)
    for h1 in range(2):
        for h2 in range(2):
            for h3 in range(2):
                for h4 in range(2):
                    for h5 in range(2):
                        for h6 in range(2):
                            for h7 in range(2):
                                for h8 in range(2):
                                    # idx = h1*2**7 + h2*2**6 + h3*2**5 + h4*2**4 + h5*2**3 + h6*2**2 + h7*2**1 + h8*2**0
                                    idx = h1*2**0 + h2*2**1 + h3*2**2 + h4*2**3 + h5*2**4 + h6*2**5 + h7*2**6 + h8*2**7
                                    theta = symbolicExecFe(Fe,h1,h2,h3,h4,h5,h6,h7,h8)
                                    # if idx==136:
                                        # print(h1,h2,h3,h4,h5,h6,h7,h8)
                                    Filters[idx] = theta
                                    #np.concatenate((thetaXX[np.newaxis], thetaXY[np.newaxis],thetaXZ[np.newaxis],thetaYX[np.newaxis],thetaYY[np.newaxis],thetaYZ[np.newaxis],thetaZX[np.newaxis],thetaZY[np.newaxis],thetaZZ[np.newaxis]), axis=0)
    return Filters         
def symbolicExecFe(Fe,h1,h2,h3,h4,h5,h6,h7,h8):
    for i in range(24):
        exec("global fe%d ; fe%d = Fe[i]"%(i+1,i+1))
    theta = np.zeros((3,),dtype = Fe.dtype)
    theta[0] = fe1*h8 + fe4*h7 + fe7*h6 + fe10*h5 + fe13*h4 + fe16*h3 + fe19*h2 + fe22*h1
    theta[1] = fe2*h8 + fe5*h7 + fe8*h6 + fe11*h5 + fe14*h4 + fe17*h3 + fe20*h2 + fe23*h1
    theta[2] = fe3*h8 + fe6*h7 + fe9*h6 + fe12*h5 + fe15*h4 + fe18*h3 + fe21*h2 + fe24*h1
    return theta

def getFilters(Ke):
    Filters = np.zeros((256,3,3,27),dtype = Ke.dtype)
    for h1 in range(2):
        for h2 in range(2):
            for h3 in range(2):
                for h4 in range(2):
                    for h5 in range(2):
                        for h6 in range(2):
                            for h7 in range(2):
                                for h8 in range(2):
                                    # idx = h1*2**7 + h2*2**6 + h3*2**5 + h4*2**4 + h5*2**3 + h6*2**2 + h7*2**1 + h8*2**0
                                    idx = h1*2**0 + h2*2**1 + h3*2**2 + h4*2**3 + h5*2**4 + h6*2**5 + h7*2**6 + h8*2**7
                                    theta = symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8)
                                    # if idx==136:
                                        # print(h1,h2,h3,h4,h5,h6,h7,h8)
                                    Filters[idx] = theta
                                    #np.concatenate((thetaXX[np.newaxis], thetaXY[np.newaxis],thetaXZ[np.newaxis],thetaYX[np.newaxis],thetaYY[np.newaxis],thetaYZ[np.newaxis],thetaZX[np.newaxis],thetaZY[np.newaxis],thetaZZ[np.newaxis]), axis=0)
    return Filters                 
def symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8):
    for i in range(24):
        for j in range(24):
            exec("global k%d_%d ; k%d_%d = Ke[i,j]"%(i+1,j+1,i+1,j+1))
    theta = np.zeros((3,3,27),dtype = Ke.dtype)
    H = np.array([h1,h2,h3,h4,h5,h6,h7,h8])
    # print(H)
    H = H.astype(np.float64)
    # print(H)
    h1 = H[0];h2 = H[1];h3 = H[2];h4 = H[3];
    h5 = H[4];h6 = H[5];h7 = H[6];h8 = H[7];
    
    # print(h1,h2,h3,h4,h5,h6,h7,h8)
    # print(h4*k13_1, h3*k16_4, h2*k19_7, h1*k22_10)
    
    theta[0,0,0]=h1*k22_1;
    theta[0,0,1]=h2*k19_1 + h1*k22_4;
    theta[0,0,2]=h2*k19_4;
    theta[0,0,3]=h3*k16_1 + h1*k22_7;
    theta[0,0,4]=h4*k13_1 + h3*k16_4 + h2*k19_7 + h1*k22_10;
    theta[0,0,5]=h4*k13_4 + h2*k19_10;
    theta[0,0,6]=h3*k16_7;
    theta[0,0,7]=h4*k13_7 + h3*k16_10;
    theta[0,0,8]=h4*k13_10;
    theta[0,0,9]=h5*k10_1 + h1*k22_13;
    theta[0,0,10]=h5*k10_4 + h2*k19_13 + h1*k22_16 + h6*k7_1;
    theta[0,0,11]=h2*k19_16 + h6*k7_4;
    theta[0,0,12]=h5*k10_7 + h3*k16_13 + h1*k22_19 + h7*k4_1;
    theta[0,0,13]=h5*k10_10 + h4*k13_13 + h3*k16_16 + h2*k19_19 + h1*k22_22 + h8*k1_1 + h7*k4_4 + h6*k7_7;
    theta[0,0,14]=h4*k13_16 + h2*k19_22 + h8*k1_4 + h6*k7_10;
    theta[0,0,15]=h3*k16_19 + h7*k4_7;
    theta[0,0,16]=h4*k13_19 + h3*k16_22 + h8*k1_7 + h7*k4_10;
    theta[0,0,17]=h4*k13_22 + h8*k1_10;
    theta[0,0,18]=h5*k10_13;
    theta[0,0,19]=h5*k10_16 + h6*k7_13;
    theta[0,0,20]=h6*k7_16;
    theta[0,0,21]=h5*k10_19 + h7*k4_13;
    theta[0,0,22]=h5*k10_22 + h8*k1_13 + h7*k4_16 + h6*k7_19;
    theta[0,0,23]=h8*k1_16 + h6*k7_22;
    theta[0,0,24]=h7*k4_19;
    theta[0,0,25]=h8*k1_19 + h7*k4_22;
    theta[0,0,26]=h8*k1_22;
    
    theta[0,1,0]=h1*k22_2;
    theta[0,1,1]=h2*k19_2 + h1*k22_5;
    theta[0,1,2]=h2*k19_5;
    theta[0,1,3]=h3*k16_2 + h1*k22_8;
    theta[0,1,4]=h4*k13_2 + h3*k16_5 + h2*k19_8 + h1*k22_11;
    theta[0,1,5]=h4*k13_5 + h2*k19_11;
    theta[0,1,6]=h3*k16_8;
    theta[0,1,7]=h4*k13_8 + h3*k16_11;
    theta[0,1,8]=h4*k13_11;
    theta[0,1,9]=h5*k10_2 + h1*k22_14;
    theta[0,1,10]=h5*k10_5 + h2*k19_14 + h1*k22_17 + h6*k7_2;
    theta[0,1,11]=h2*k19_17 + h6*k7_5;
    theta[0,1,12]=h5*k10_8 + h3*k16_14 + h1*k22_20 + h7*k4_2;
    theta[0,1,13]=h5*k10_11 + h4*k13_14 + h3*k16_17 + h2*k19_20 + h1*k22_23 + h8*k1_2 + h7*k4_5 + h6*k7_8;
    theta[0,1,14]=h4*k13_17 + h2*k19_23 + h8*k1_5 + h6*k7_11;
    theta[0,1,15]=h3*k16_20 + h7*k4_8;
    theta[0,1,16]=h4*k13_20 + h3*k16_23 + h8*k1_8 + h7*k4_11;
    theta[0,1,17]=h4*k13_23 + h8*k1_11;
    theta[0,1,18]=h5*k10_14;
    theta[0,1,19]=h5*k10_17 + h6*k7_14;
    theta[0,1,20]=h6*k7_17;
    theta[0,1,21]=h5*k10_20 + h7*k4_14;
    theta[0,1,22]=h5*k10_23 + h8*k1_14 + h7*k4_17 + h6*k7_20;
    theta[0,1,23]=h8*k1_17 + h6*k7_23;
    theta[0,1,24]=h7*k4_20;
    theta[0,1,25]=h8*k1_20 + h7*k4_23;
    theta[0,1,26]=h8*k1_23;
    
    theta[0,2,0]=h1*k22_3;
    theta[0,2,1]=h2*k19_3 + h1*k22_6;
    theta[0,2,2]=h2*k19_6;
    theta[0,2,3]=h3*k16_3 + h1*k22_9;
    theta[0,2,4]=h4*k13_3 + h3*k16_6 + h2*k19_9 + h1*k22_12;
    theta[0,2,5]=h4*k13_6 + h2*k19_12;
    theta[0,2,6]=h3*k16_9;
    theta[0,2,7]=h4*k13_9 + h3*k16_12;
    theta[0,2,8]=h4*k13_12;
    theta[0,2,9]=h5*k10_3 + h1*k22_15;
    theta[0,2,10]=h5*k10_6 + h2*k19_15 + h1*k22_18 + h6*k7_3;
    theta[0,2,11]=h2*k19_18 + h6*k7_6;
    theta[0,2,12]=h5*k10_9 + h3*k16_15 + h1*k22_21 + h7*k4_3;
    theta[0,2,13]=h5*k10_12 + h4*k13_15 + h3*k16_18 + h2*k19_21 + h1*k22_24 + h8*k1_3 + h7*k4_6 + h6*k7_9;
    theta[0,2,14]=h4*k13_18 + h2*k19_24 + h8*k1_6 + h6*k7_12;
    theta[0,2,15]=h3*k16_21 + h7*k4_9;
    theta[0,2,16]=h4*k13_21 + h3*k16_24 + h8*k1_9 + h7*k4_12;
    theta[0,2,17]=h4*k13_24 + h8*k1_12;
    theta[0,2,18]=h5*k10_15;
    theta[0,2,19]=h5*k10_18 + h6*k7_15;
    theta[0,2,20]=h6*k7_18;
    theta[0,2,21]=h5*k10_21 + h7*k4_15;
    theta[0,2,22]=h5*k10_24 + h8*k1_15 + h7*k4_18 + h6*k7_21;
    theta[0,2,23]=h8*k1_18 + h6*k7_24;
    theta[0,2,24]=h7*k4_21;
    theta[0,2,25]=h8*k1_21 + h7*k4_24;
    theta[0,2,26]=h8*k1_24;

    theta[1,0,0]=h1*k23_1;
    theta[1,0,1]=h2*k20_1 + h1*k23_4;
    theta[1,0,2]=h2*k20_4;
    theta[1,0,3]=h3*k17_1 + h1*k23_7;
    theta[1,0,4]=h4*k14_1 + h3*k17_4 + h2*k20_7 + h1*k23_10;
    theta[1,0,5]=h4*k14_4 + h2*k20_10;
    theta[1,0,6]=h3*k17_7;
    theta[1,0,7]=h4*k14_7 + h3*k17_10;
    theta[1,0,8]=h4*k14_10;
    theta[1,0,9]=h5*k11_1 + h1*k23_13;
    theta[1,0,10]=h5*k11_4 + h2*k20_13 + h1*k23_16 + h6*k8_1;
    theta[1,0,11]=h2*k20_16 + h6*k8_4;
    theta[1,0,12]=h5*k11_7 + h3*k17_13 + h1*k23_19 + h7*k5_1;
    theta[1,0,13]=h5*k11_10 + h4*k14_13 + h3*k17_16 + h2*k20_19 + h1*k23_22 + h8*k2_1 + h7*k5_4 + h6*k8_7;
    theta[1,0,14]=h4*k14_16 + h2*k20_22 + h8*k2_4 + h6*k8_10;
    theta[1,0,15]=h3*k17_19 + h7*k5_7;
    theta[1,0,16]=h4*k14_19 + h3*k17_22 + h8*k2_7 + h7*k5_10;
    theta[1,0,17]=h4*k14_22 + h8*k2_10;
    theta[1,0,18]=h5*k11_13;
    theta[1,0,19]=h5*k11_16 + h6*k8_13;
    theta[1,0,20]=h6*k8_16;
    theta[1,0,21]=h5*k11_19 + h7*k5_13;
    theta[1,0,22]=h5*k11_22 + h8*k2_13 + h7*k5_16 + h6*k8_19;
    theta[1,0,23]=h8*k2_16 + h6*k8_22;
    theta[1,0,24]=h7*k5_19;
    theta[1,0,25]=h8*k2_19 + h7*k5_22;
    theta[1,0,26]=h8*k2_22;
    
    theta[1,1,0]=h1*k23_2;
    theta[1,1,1]=h2*k20_2 + h1*k23_5;
    theta[1,1,2]=h2*k20_5;
    theta[1,1,3]=h3*k17_2 + h1*k23_8;
    theta[1,1,4]=h4*k14_2 + h3*k17_5 + h2*k20_8 + h1*k23_11;
    theta[1,1,5]=h4*k14_5 + h2*k20_11;
    theta[1,1,6]=h3*k17_8;
    theta[1,1,7]=h4*k14_8 + h3*k17_11;
    theta[1,1,8]=h4*k14_11;
    theta[1,1,9]=h5*k11_2 + h1*k23_14;
    theta[1,1,10]=h5*k11_5 + h2*k20_14 + h1*k23_17 + h6*k8_2;
    theta[1,1,11]=h2*k20_17 + h6*k8_5;
    theta[1,1,12]=h5*k11_8 + h3*k17_14 + h1*k23_20 + h7*k5_2;
    theta[1,1,13]=h5*k11_11 + h4*k14_14 + h3*k17_17 + h2*k20_20 + h1*k23_23 + h8*k2_2 + h7*k5_5 + h6*k8_8;
    theta[1,1,14]=h4*k14_17 + h2*k20_23 + h8*k2_5 + h6*k8_11;
    theta[1,1,15]=h3*k17_20 + h7*k5_8;
    theta[1,1,16]=h4*k14_20 + h3*k17_23 + h8*k2_8 + h7*k5_11;
    theta[1,1,17]=h4*k14_23 + h8*k2_11;
    theta[1,1,18]=h5*k11_14;
    theta[1,1,19]=h5*k11_17 + h6*k8_14;
    theta[1,1,20]=h6*k8_17;
    theta[1,1,21]=h5*k11_20 + h7*k5_14;
    theta[1,1,22]=h5*k11_23 + h8*k2_14 + h7*k5_17 + h6*k8_20;
    theta[1,1,23]=h8*k2_17 + h6*k8_23;
    theta[1,1,24]=h7*k5_20;
    theta[1,1,25]=h8*k2_20 + h7*k5_23;
    theta[1,1,26]=h8*k2_23;
    
    theta[1,2,0]=h1*k23_3;
    theta[1,2,1]=h2*k20_3 + h1*k23_6;
    theta[1,2,2]=h2*k20_6;
    theta[1,2,3]=h3*k17_3 + h1*k23_9;
    theta[1,2,4]=h4*k14_3 + h3*k17_6 + h2*k20_9 + h1*k23_12;
    theta[1,2,5]=h4*k14_6 + h2*k20_12;
    theta[1,2,6]=h3*k17_9;
    theta[1,2,7]=h4*k14_9 + h3*k17_12;
    theta[1,2,8]=h4*k14_12;
    theta[1,2,9]=h5*k11_3 + h1*k23_15;
    theta[1,2,10]=h5*k11_6 + h2*k20_15 + h1*k23_18 + h6*k8_3;
    theta[1,2,11]=h2*k20_18 + h6*k8_6;
    theta[1,2,12]=h5*k11_9 + h3*k17_15 + h1*k23_21 + h7*k5_3;
    theta[1,2,13]=h5*k11_12 + h4*k14_15 + h3*k17_18 + h2*k20_21 + h1*k23_24 + h8*k2_3 + h7*k5_6 + h6*k8_9;
    theta[1,2,14]=h4*k14_18 + h2*k20_24 + h8*k2_6 + h6*k8_12;
    theta[1,2,15]=h3*k17_21 + h7*k5_9;
    theta[1,2,16]=h4*k14_21 + h3*k17_24 + h8*k2_9 + h7*k5_12;
    theta[1,2,17]=h4*k14_24 + h8*k2_12;
    theta[1,2,18]=h5*k11_15;
    theta[1,2,19]=h5*k11_18 + h6*k8_15;
    theta[1,2,20]=h6*k8_18;
    theta[1,2,21]=h5*k11_21 + h7*k5_15;
    theta[1,2,22]=h5*k11_24 + h8*k2_15 + h7*k5_18 + h6*k8_21;
    theta[1,2,23]=h8*k2_18 + h6*k8_24;
    theta[1,2,24]=h7*k5_21;
    theta[1,2,25]=h8*k2_21 + h7*k5_24;
    theta[1,2,26]=h8*k2_24;

    theta[2,0,0]=h1*k24_1;
    theta[2,0,1]=h2*k21_1 + h1*k24_4;
    theta[2,0,2]=h2*k21_4;
    theta[2,0,3]=h3*k18_1 + h1*k24_7;
    theta[2,0,4]=h4*k15_1 + h3*k18_4 + h2*k21_7 + h1*k24_10;
    theta[2,0,5]=h4*k15_4 + h2*k21_10;
    theta[2,0,6]=h3*k18_7;
    theta[2,0,7]=h4*k15_7 + h3*k18_10;
    theta[2,0,8]=h4*k15_10;
    theta[2,0,9]=h5*k12_1 + h1*k24_13;
    theta[2,0,10]=h5*k12_4 + h2*k21_13 + h1*k24_16 + h6*k9_1;
    theta[2,0,11]=h2*k21_16 + h6*k9_4;
    theta[2,0,12]=h5*k12_7 + h3*k18_13 + h1*k24_19 + h7*k6_1;
    theta[2,0,13]=h5*k12_10 + h4*k15_13 + h3*k18_16 + h2*k21_19 + h1*k24_22 + h8*k3_1 + h7*k6_4 + h6*k9_7;
    theta[2,0,14]=h4*k15_16 + h2*k21_22 + h8*k3_4 + h6*k9_10;
    theta[2,0,15]=h3*k18_19 + h7*k6_7;
    theta[2,0,16]=h4*k15_19 + h3*k18_22 + h8*k3_7 + h7*k6_10;
    theta[2,0,17]=h4*k15_22 + h8*k3_10;
    theta[2,0,18]=h5*k12_13;
    theta[2,0,19]=h5*k12_16 + h6*k9_13;
    theta[2,0,20]=h6*k9_16;
    theta[2,0,21]=h5*k12_19 + h7*k6_13;
    theta[2,0,22]=h5*k12_22 + h8*k3_13 + h7*k6_16 + h6*k9_19;
    theta[2,0,23]=h8*k3_16 + h6*k9_22;
    theta[2,0,24]=h7*k6_19;
    theta[2,0,25]=h8*k3_19 + h7*k6_22;
    theta[2,0,26]=h8*k3_22;
    
    theta[2,1,0]=h1*k24_2;
    theta[2,1,1]=h2*k21_2 + h1*k24_5;
    theta[2,1,2]=h2*k21_5;
    theta[2,1,3]=h3*k18_2 + h1*k24_8;
    theta[2,1,4]=h4*k15_2 + h3*k18_5 + h2*k21_8 + h1*k24_11;
    theta[2,1,5]=h4*k15_5 + h2*k21_11;
    theta[2,1,6]=h3*k18_8;
    theta[2,1,7]=h4*k15_8 + h3*k18_11;
    theta[2,1,8]=h4*k15_11;
    theta[2,1,9]=h5*k12_2 + h1*k24_14;
    theta[2,1,10]=h5*k12_5 + h2*k21_14 + h1*k24_17 + h6*k9_2;
    theta[2,1,11]=h2*k21_17 + h6*k9_5;
    theta[2,1,12]=h5*k12_8 + h3*k18_14 + h1*k24_20 + h7*k6_2;
    theta[2,1,13]=h5*k12_11 + h4*k15_14 + h3*k18_17 + h2*k21_20 + h1*k24_23 + h8*k3_2 + h7*k6_5 + h6*k9_8;
    theta[2,1,14]=h4*k15_17 + h2*k21_23 + h8*k3_5 + h6*k9_11;
    theta[2,1,15]=h3*k18_20 + h7*k6_8;
    theta[2,1,16]=h4*k15_20 + h3*k18_23 + h8*k3_8 + h7*k6_11;
    theta[2,1,17]=h4*k15_23 + h8*k3_11;
    theta[2,1,18]=h5*k12_14;
    theta[2,1,19]=h5*k12_17 + h6*k9_14;
    theta[2,1,20]=h6*k9_17;
    theta[2,1,21]=h5*k12_20 + h7*k6_14;
    theta[2,1,22]=h5*k12_23 + h8*k3_14 + h7*k6_17 + h6*k9_20;
    theta[2,1,23]=h8*k3_17 + h6*k9_23;
    theta[2,1,24]=h7*k6_20;
    theta[2,1,25]=h8*k3_20 + h7*k6_23;
    theta[2,1,26]=h8*k3_23;
    
    theta[2,2,0]=h1*k24_3;
    theta[2,2,1]=h2*k21_3 + h1*k24_6;
    theta[2,2,2]=h2*k21_6;
    theta[2,2,3]=h3*k18_3 + h1*k24_9;
    theta[2,2,4]=h4*k15_3 + h3*k18_6 + h2*k21_9 + h1*k24_12;
    theta[2,2,5]=h4*k15_6 + h2*k21_12;
    theta[2,2,6]=h3*k18_9;
    theta[2,2,7]=h4*k15_9 + h3*k18_12;
    theta[2,2,8]=h4*k15_12;
    theta[2,2,9]=h5*k12_3 + h1*k24_15;
    theta[2,2,10]=h5*k12_6 + h2*k21_15 + h1*k24_18 + h6*k9_3;
    theta[2,2,11]=h2*k21_18 + h6*k9_6;
    theta[2,2,12]=h5*k12_9 + h3*k18_15 + h1*k24_21 + h7*k6_3;
    theta[2,2,13]=h5*k12_12 + h4*k15_15 + h3*k18_18 + h2*k21_21 + h1*k24_24 + h8*k3_3 + h7*k6_6 + h6*k9_9;
    theta[2,2,14]=h4*k15_18 + h2*k21_24 + h8*k3_6 + h6*k9_12;
    theta[2,2,15]=h3*k18_21 + h7*k6_9;
    theta[2,2,16]=h4*k15_21 + h3*k18_24 + h8*k3_9 + h7*k6_12;
    theta[2,2,17]=h4*k15_24 + h8*k3_12;
    theta[2,2,18]=h5*k12_15;
    theta[2,2,19]=h5*k12_18 + h6*k9_15;
    theta[2,2,20]=h6*k9_18;
    theta[2,2,21]=h5*k12_21 + h7*k6_15;
    theta[2,2,22]=h5*k12_24 + h8*k3_15 + h7*k6_18 + h6*k9_21;
    theta[2,2,23]=h8*k3_18 + h6*k9_24;
    theta[2,2,24]=h7*k6_21;
    theta[2,2,25]=h8*k3_21 + h7*k6_24;
    theta[2,2,26]=h8*k3_24;

    return theta

if __name__ == "__main__":
    Ke = np.eye(24,dtype = np.float64)
    # Ke = np.ones((24,24))
    Ke[0,1] = 1
    
    resolution = 40 
    E = 1e6; nu = 0.3
    from ConstMatricesForHomogenization import ISOelasticitytensor,LocalKeFe
    D0 = ISOelasticitytensor(E, nu)
    Ke,Fe,intB = LocalKeFe(resolution,D0)
    Ke = Ke.astype(np.float64)
    Filters = getFilters(Ke)
    # h1=1;h2=1;h3=1;h4=1;h5=1;h6=1;h7=1;h8=1;
    # thetaXX,thetaXY,thetaXZ,thetaYX,thetaYY,thetaYZ,thetaZX,thetaZY,thetaZZ = symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8)