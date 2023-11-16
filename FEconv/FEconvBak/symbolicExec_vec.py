# -*- coding: utf-8 -*-
"""
Created on Fri Feb 5 21:00:23 2021

@author: Liangchao Zhu
"""

import numpy as np

def getFilters(Ke):
    Filters = np.zeros((256,9,27))
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
                                    thetaXX,thetaXY,thetaXZ,thetaYX,thetaYY,thetaYZ,thetaZX,thetaZY,thetaZZ = symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8)
                                    Filters[idx] = np.concatenate((thetaXX[np.newaxis], thetaXY[np.newaxis],thetaXZ[np.newaxis],thetaYX[np.newaxis],thetaYY[np.newaxis],thetaYZ[np.newaxis],thetaZX[np.newaxis],thetaZY[np.newaxis],thetaZZ[np.newaxis]), axis=0)
    return Filters                 
def symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8):
    for i in range(24):
        for j in range(24):
            exec("global k%d_%d ; k%d_%d = Ke[i,j]"%(i+1,j+1,i+1,j+1))
    thetaXX = np.zeros((27))
    thetaXY = np.zeros((27))
    thetaXZ = np.zeros((27))
    
    thetaYX = np.zeros((27))
    thetaYY = np.zeros((27))
    thetaYZ = np.zeros((27))
    
    thetaZX = np.zeros((27))
    thetaZY = np.zeros((27))
    thetaZZ = np.zeros((27))
    
    thetaXX[0]=h1*k24_1;
    thetaXX[1]=h2*k21_1 + h1*k24_4;
    thetaXX[2]=h2*k21_4;
    thetaXX[3]=h3*k18_1 + h1*k24_7;
    thetaXX[4]=h4*k15_1 + h3*k18_4 + h2*k21_7 + h1*k24_10;
    thetaXX[5]=h4*k15_4 + h2*k21_10;
    thetaXX[6]=h3*k18_7;
    thetaXX[7]=h4*k15_7 + h3*k18_10;
    thetaXX[8]=h4*k15_10;
    thetaXX[9]=h5*k12_1 + h1*k24_13;
    thetaXX[10]=h5*k12_4 + h2*k21_13 + h1*k24_16 + h6*k9_1;
    thetaXX[11]=h2*k21_16 + h6*k9_4;
    thetaXX[12]=h5*k12_7 + h3*k18_13 + h1*k24_19 + h7*k6_1;
    thetaXX[13]=h5*k12_10 + h4*k15_13 + h3*k18_16 + h2*k21_19 + h1*k24_22 + h8*k3_1 + h7*k6_4 + h6*k9_7;
    thetaXX[14]=h4*k15_16 + h2*k21_22 + h8*k3_4 + h6*k9_10;
    thetaXX[15]=h3*k18_19 + h7*k6_7;
    thetaXX[16]=h4*k15_19 + h3*k18_22 + h8*k3_7 + h7*k6_10;
    thetaXX[17]=h4*k15_22 + h8*k3_10;
    thetaXX[18]=h5*k12_13;
    thetaXX[19]=h5*k12_16 + h6*k9_13;
    thetaXX[20]=h6*k9_16;
    thetaXX[21]=h5*k12_19 + h7*k6_13;
    thetaXX[22]=h5*k12_22 + h8*k3_13 + h7*k6_16 + h6*k9_19;
    thetaXX[23]=h8*k3_16 + h6*k9_22;
    thetaXX[24]=h7*k6_19;
    thetaXX[25]=h8*k3_19 + h7*k6_22;
    thetaXX[26]=h8*k3_22;
    
    thetaXY[0]=h1*k24_2;
    thetaXY[1]=h2*k21_2 + h1*k24_5;
    thetaXY[2]=h2*k21_5;
    thetaXY[3]=h3*k18_2 + h1*k24_8;
    thetaXY[4]=h4*k15_2 + h3*k18_5 + h2*k21_8 + h1*k24_11;
    thetaXY[5]=h4*k15_5 + h2*k21_11;
    thetaXY[6]=h3*k18_8;
    thetaXY[7]=h4*k15_8 + h3*k18_11;
    thetaXY[8]=h4*k15_11;
    thetaXY[9]=h5*k12_2 + h1*k24_14;
    thetaXY[10]=h5*k12_5 + h2*k21_14 + h1*k24_17 + h6*k9_2;
    thetaXY[11]=h2*k21_17 + h6*k9_5;
    thetaXY[12]=h5*k12_8 + h3*k18_14 + h1*k24_20 + h7*k6_2;
    thetaXY[13]=h5*k12_11 + h4*k15_14 + h3*k18_17 + h2*k21_20 + h1*k24_23 + h8*k3_2 + h7*k6_5 + h6*k9_8;
    thetaXY[14]=h4*k15_17 + h2*k21_23 + h8*k3_5 + h6*k9_11;
    thetaXY[15]=h3*k18_20 + h7*k6_8;
    thetaXY[16]=h4*k15_20 + h3*k18_23 + h8*k3_8 + h7*k6_11;
    thetaXY[17]=h4*k15_23 + h8*k3_11;
    thetaXY[18]=h5*k12_14;
    thetaXY[19]=h5*k12_17 + h6*k9_14;
    thetaXY[20]=h6*k9_17;
    thetaXY[21]=h5*k12_20 + h7*k6_14;
    thetaXY[22]=h5*k12_23 + h8*k3_14 + h7*k6_17 + h6*k9_20;
    thetaXY[23]=h8*k3_17 + h6*k9_23;
    thetaXY[24]=h7*k6_20;
    thetaXY[25]=h8*k3_20 + h7*k6_23;
    thetaXY[26]=h8*k3_23;
    
    thetaXZ[0]=h1*k24_3;
    thetaXZ[1]=h2*k21_3 + h1*k24_6;
    thetaXZ[2]=h2*k21_6;
    thetaXZ[3]=h3*k18_3 + h1*k24_9;
    thetaXZ[4]=h4*k15_3 + h3*k18_6 + h2*k21_9 + h1*k24_12;
    thetaXZ[5]=h4*k15_6 + h2*k21_12;
    thetaXZ[6]=h3*k18_9;
    thetaXZ[7]=h4*k15_9 + h3*k18_12;
    thetaXZ[8]=h4*k15_12;
    thetaXZ[9]=h5*k12_3 + h1*k24_15;
    thetaXZ[10]=h5*k12_6 + h2*k21_15 + h1*k24_18 + h6*k9_3;
    thetaXZ[11]=h2*k21_18 + h6*k9_6;
    thetaXZ[12]=h5*k12_9 + h3*k18_15 + h1*k24_21 + h7*k6_3;
    thetaXZ[13]=h5*k12_12 + h4*k15_15 + h3*k18_18 + h2*k21_21 + h1*k24_24 + h8*k3_3 + h7*k6_6 + h6*k9_9;
    thetaXZ[14]=h4*k15_18 + h2*k21_24 + h8*k3_6 + h6*k9_12;
    thetaXZ[15]=h3*k18_21 + h7*k6_9;
    thetaXZ[16]=h4*k15_21 + h3*k18_24 + h8*k3_9 + h7*k6_12;
    thetaXZ[17]=h4*k15_24 + h8*k3_12;
    thetaXZ[18]=h5*k12_15;
    thetaXZ[19]=h5*k12_18 + h6*k9_15;
    thetaXZ[20]=h6*k9_18;
    thetaXZ[21]=h5*k12_21 + h7*k6_15;
    thetaXZ[22]=h5*k12_24 + h8*k3_15 + h7*k6_18 + h6*k9_21;
    thetaXZ[23]=h8*k3_18 + h6*k9_24;
    thetaXZ[24]=h7*k6_21;
    thetaXZ[25]=h8*k3_21 + h7*k6_24;
    thetaXZ[26]=h8*k3_24;
    
    thetaYX[0]=h1*k24_1;
    thetaYX[1]=h2*k21_1 + h1*k24_4;
    thetaYX[2]=h2*k21_4;
    thetaYX[3]=h3*k18_1 + h1*k24_7;
    thetaYX[4]=h4*k15_1 + h3*k18_4 + h2*k21_7 + h1*k24_10;
    thetaYX[5]=h4*k15_4 + h2*k21_10;
    thetaYX[6]=h3*k18_7;
    thetaYX[7]=h4*k15_7 + h3*k18_10;
    thetaYX[8]=h4*k15_10;
    thetaYX[9]=h5*k12_1 + h1*k24_13;
    thetaYX[10]=h5*k12_4 + h2*k21_13 + h1*k24_16 + h6*k9_1;
    thetaYX[11]=h2*k21_16 + h6*k9_4;
    thetaYX[12]=h5*k12_7 + h3*k18_13 + h1*k24_19 + h7*k6_1;
    thetaYX[13]=h5*k12_10 + h4*k15_13 + h3*k18_16 + h2*k21_19 + h1*k24_22 + h8*k3_1 + h7*k6_4 + h6*k9_7;
    thetaYX[14]=h4*k15_16 + h2*k21_22 + h8*k3_4 + h6*k9_10;
    thetaYX[15]=h3*k18_19 + h7*k6_7;
    thetaYX[16]=h4*k15_19 + h3*k18_22 + h8*k3_7 + h7*k6_10;
    thetaYX[17]=h4*k15_22 + h8*k3_10;
    thetaYX[18]=h5*k12_13;
    thetaYX[19]=h5*k12_16 + h6*k9_13;
    thetaYX[20]=h6*k9_16;
    thetaYX[21]=h5*k12_19 + h7*k6_13;
    thetaYX[22]=h5*k12_22 + h8*k3_13 + h7*k6_16 + h6*k9_19;
    thetaYX[23]=h8*k3_16 + h6*k9_22;
    thetaYX[24]=h7*k6_19;
    thetaYX[25]=h8*k3_19 + h7*k6_22;
    thetaYX[26]=h8*k3_22;
    
    thetaYY[0]=h1*k24_2;
    thetaYY[1]=h2*k21_2 + h1*k24_5;
    thetaYY[2]=h2*k21_5;
    thetaYY[3]=h3*k18_2 + h1*k24_8;
    thetaYY[4]=h4*k15_2 + h3*k18_5 + h2*k21_8 + h1*k24_11;
    thetaYY[5]=h4*k15_5 + h2*k21_11;
    thetaYY[6]=h3*k18_8;
    thetaYY[7]=h4*k15_8 + h3*k18_11;
    thetaYY[8]=h4*k15_11;
    thetaYY[9]=h5*k12_2 + h1*k24_14;
    thetaYY[10]=h5*k12_5 + h2*k21_14 + h1*k24_17 + h6*k9_2;
    thetaYY[11]=h2*k21_17 + h6*k9_5;
    thetaYY[12]=h5*k12_8 + h3*k18_14 + h1*k24_20 + h7*k6_2;
    thetaYY[13]=h5*k12_11 + h4*k15_14 + h3*k18_17 + h2*k21_20 + h1*k24_23 + h8*k3_2 + h7*k6_5 + h6*k9_8;
    thetaYY[14]=h4*k15_17 + h2*k21_23 + h8*k3_5 + h6*k9_11;
    thetaYY[15]=h3*k18_20 + h7*k6_8;
    thetaYY[16]=h4*k15_20 + h3*k18_23 + h8*k3_8 + h7*k6_11;
    thetaYY[17]=h4*k15_23 + h8*k3_11;
    thetaYY[18]=h5*k12_14;
    thetaYY[19]=h5*k12_17 + h6*k9_14;
    thetaYY[20]=h6*k9_17;
    thetaYY[21]=h5*k12_20 + h7*k6_14;
    thetaYY[22]=h5*k12_23 + h8*k3_14 + h7*k6_17 + h6*k9_20;
    thetaYY[23]=h8*k3_17 + h6*k9_23;
    thetaYY[24]=h7*k6_20;
    thetaYY[25]=h8*k3_20 + h7*k6_23;
    thetaYY[26]=h8*k3_23;
    
    thetaYZ[0]=h1*k24_3;
    thetaYZ[1]=h2*k21_3 + h1*k24_6;
    thetaYZ[2]=h2*k21_6;
    thetaYZ[3]=h3*k18_3 + h1*k24_9;
    thetaYZ[4]=h4*k15_3 + h3*k18_6 + h2*k21_9 + h1*k24_12;
    thetaYZ[5]=h4*k15_6 + h2*k21_12;
    thetaYZ[6]=h3*k18_9;
    thetaYZ[7]=h4*k15_9 + h3*k18_12;
    thetaYZ[8]=h4*k15_12;
    thetaYZ[9]=h5*k12_3 + h1*k24_15;
    thetaYZ[10]=h5*k12_6 + h2*k21_15 + h1*k24_18 + h6*k9_3;
    thetaYZ[11]=h2*k21_18 + h6*k9_6;
    thetaYZ[12]=h5*k12_9 + h3*k18_15 + h1*k24_21 + h7*k6_3;
    thetaYZ[13]=h5*k12_12 + h4*k15_15 + h3*k18_18 + h2*k21_21 + h1*k24_24 + h8*k3_3 + h7*k6_6 + h6*k9_9;
    thetaYZ[14]=h4*k15_18 + h2*k21_24 + h8*k3_6 + h6*k9_12;
    thetaYZ[15]=h3*k18_21 + h7*k6_9;
    thetaYZ[16]=h4*k15_21 + h3*k18_24 + h8*k3_9 + h7*k6_12;
    thetaYZ[17]=h4*k15_24 + h8*k3_12;
    thetaYZ[18]=h5*k12_15;
    thetaYZ[19]=h5*k12_18 + h6*k9_15;
    thetaYZ[20]=h6*k9_18;
    thetaYZ[21]=h5*k12_21 + h7*k6_15;
    thetaYZ[22]=h5*k12_24 + h8*k3_15 + h7*k6_18 + h6*k9_21;
    thetaYZ[23]=h8*k3_18 + h6*k9_24;
    thetaYZ[24]=h7*k6_21;
    thetaYZ[25]=h8*k3_21 + h7*k6_24;
    thetaYZ[26]=h8*k3_24;
    
    thetaZX[0]=h1*k24_1;
    thetaZX[1]=h2*k21_1 + h1*k24_4;
    thetaZX[2]=h2*k21_4;
    thetaZX[3]=h3*k18_1 + h1*k24_7;
    thetaZX[4]=h4*k15_1 + h3*k18_4 + h2*k21_7 + h1*k24_10;
    thetaZX[5]=h4*k15_4 + h2*k21_10;
    thetaZX[6]=h3*k18_7;
    thetaZX[7]=h4*k15_7 + h3*k18_10;
    thetaZX[8]=h4*k15_10;
    thetaZX[9]=h5*k12_1 + h1*k24_13;
    thetaZX[10]=h5*k12_4 + h2*k21_13 + h1*k24_16 + h6*k9_1;
    thetaZX[11]=h2*k21_16 + h6*k9_4;
    thetaZX[12]=h5*k12_7 + h3*k18_13 + h1*k24_19 + h7*k6_1;
    thetaZX[13]=h5*k12_10 + h4*k15_13 + h3*k18_16 + h2*k21_19 + h1*k24_22 + h8*k3_1 + h7*k6_4 + h6*k9_7;
    thetaZX[14]=h4*k15_16 + h2*k21_22 + h8*k3_4 + h6*k9_10;
    thetaZX[15]=h3*k18_19 + h7*k6_7;
    thetaZX[16]=h4*k15_19 + h3*k18_22 + h8*k3_7 + h7*k6_10;
    thetaZX[17]=h4*k15_22 + h8*k3_10;
    thetaZX[18]=h5*k12_13;
    thetaZX[19]=h5*k12_16 + h6*k9_13;
    thetaZX[20]=h6*k9_16;
    thetaZX[21]=h5*k12_19 + h7*k6_13;
    thetaZX[22]=h5*k12_22 + h8*k3_13 + h7*k6_16 + h6*k9_19;
    thetaZX[23]=h8*k3_16 + h6*k9_22;
    thetaZX[24]=h7*k6_19;
    thetaZX[25]=h8*k3_19 + h7*k6_22;
    thetaZX[26]=h8*k3_22;
    
    thetaZY[0]=h1*k24_2;
    thetaZY[1]=h2*k21_2 + h1*k24_5;
    thetaZY[2]=h2*k21_5;
    thetaZY[3]=h3*k18_2 + h1*k24_8;
    thetaZY[4]=h4*k15_2 + h3*k18_5 + h2*k21_8 + h1*k24_11;
    thetaZY[5]=h4*k15_5 + h2*k21_11;
    thetaZY[6]=h3*k18_8;
    thetaZY[7]=h4*k15_8 + h3*k18_11;
    thetaZY[8]=h4*k15_11;
    thetaZY[9]=h5*k12_2 + h1*k24_14;
    thetaZY[10]=h5*k12_5 + h2*k21_14 + h1*k24_17 + h6*k9_2;
    thetaZY[11]=h2*k21_17 + h6*k9_5;
    thetaZY[12]=h5*k12_8 + h3*k18_14 + h1*k24_20 + h7*k6_2;
    thetaZY[13]=h5*k12_11 + h4*k15_14 + h3*k18_17 + h2*k21_20 + h1*k24_23 + h8*k3_2 + h7*k6_5 + h6*k9_8;
    thetaZY[14]=h4*k15_17 + h2*k21_23 + h8*k3_5 + h6*k9_11;
    thetaZY[15]=h3*k18_20 + h7*k6_8;
    thetaZY[16]=h4*k15_20 + h3*k18_23 + h8*k3_8 + h7*k6_11;
    thetaZY[17]=h4*k15_23 + h8*k3_11;
    thetaZY[18]=h5*k12_14;
    thetaZY[19]=h5*k12_17 + h6*k9_14;
    thetaZY[20]=h6*k9_17;
    thetaZY[21]=h5*k12_20 + h7*k6_14;
    thetaZY[22]=h5*k12_23 + h8*k3_14 + h7*k6_17 + h6*k9_20;
    thetaZY[23]=h8*k3_17 + h6*k9_23;
    thetaZY[24]=h7*k6_20;
    thetaZY[25]=h8*k3_20 + h7*k6_23;
    thetaZY[26]=h8*k3_23;
    
    thetaZZ[0]=h1*k24_3;
    thetaZZ[1]=h2*k21_3 + h1*k24_6;
    thetaZZ[2]=h2*k21_6;
    thetaZZ[3]=h3*k18_3 + h1*k24_9;
    thetaZZ[4]=h4*k15_3 + h3*k18_6 + h2*k21_9 + h1*k24_12;
    thetaZZ[5]=h4*k15_6 + h2*k21_12;
    thetaZZ[6]=h3*k18_9;
    thetaZZ[7]=h4*k15_9 + h3*k18_12;
    thetaZZ[8]=h4*k15_12;
    thetaZZ[9]=h5*k12_3 + h1*k24_15;
    thetaZZ[10]=h5*k12_6 + h2*k21_15 + h1*k24_18 + h6*k9_3;
    thetaZZ[11]=h2*k21_18 + h6*k9_6;
    thetaZZ[12]=h5*k12_9 + h3*k18_15 + h1*k24_21 + h7*k6_3;
    thetaZZ[13]=h5*k12_12 + h4*k15_15 + h3*k18_18 + h2*k21_21 + h1*k24_24 + h8*k3_3 + h7*k6_6 + h6*k9_9;
    thetaZZ[14]=h4*k15_18 + h2*k21_24 + h8*k3_6 + h6*k9_12;
    thetaZZ[15]=h3*k18_21 + h7*k6_9;
    thetaZZ[16]=h4*k15_21 + h3*k18_24 + h8*k3_9 + h7*k6_12;
    thetaZZ[17]=h4*k15_24 + h8*k3_12;
    thetaZZ[18]=h5*k12_15;
    thetaZZ[19]=h5*k12_18 + h6*k9_15;
    thetaZZ[20]=h6*k9_18;
    thetaZZ[21]=h5*k12_21 + h7*k6_15;
    thetaZZ[22]=h5*k12_24 + h8*k3_15 + h7*k6_18 + h6*k9_21;
    thetaZZ[23]=h8*k3_18 + h6*k9_24;
    thetaZZ[24]=h7*k6_21;
    thetaZZ[25]=h8*k3_21 + h7*k6_24;
    thetaZZ[26]=h8*k3_24;

    return thetaXX,thetaXY,thetaXZ,thetaYX,thetaYY,thetaYZ,thetaZX,thetaZY,thetaZZ

if __name__ == "__main__":
    Ke = np.eye(24)
    Filters = getFilters(Ke)
    # h1=1;h2=1;h3=1;h4=1;h5=1;h6=1;h7=1;h8=1;
    # thetaXX,thetaXY,thetaXZ,thetaYX,thetaYY,thetaYZ,thetaZX,thetaZY,thetaZZ = symbolicExec(Ke,h1,h2,h3,h4,h5,h6,h7,h8)