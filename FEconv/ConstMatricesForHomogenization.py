# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:57:00 2020

@author: Liangchao Zhu
"""
import numpy as np

def Jacobian(nodes, eles, dshapedr, dshapeds, dshapedt, eleNum):
    detjacob = np.zeros((eleNum, 8), dtype=np.float32)  # area
    invjacob = np.zeros((eleNum, 72), dtype=np.float32);
    for i in range(eleNum):
        node = eles[i, :]
        coord = nodes[node, :]  # 8 x 3

        for j in range(8):  # for 8 gauss points
            dNdr = dshapedr[j, :]
            dNds = dshapeds[j, :]
            dNdt = dshapedt[j, :]

            jacob = np.array([[dNdr @ coord[:, 0], dNdr @ coord[:, 1], dNdr @ coord[:, 2]],
                              [dNds @ coord[:, 0], dNds @ coord[:, 1], dNds @ coord[:, 2]],
                              [dNdt @ coord[:, 0], dNdt @ coord[:, 1], dNdt @ coord[:, 2]]])

            detjacob[i, j] = np.linalg.det(jacob)
            invjacob[i, j * 9:(j + 1) * 9] = np.linalg.inv(jacob).flatten()  # !!!!!!!!
            # if i==0 and j==0:
            #     print(jacob)
            #     print(np.linalg.inv(jacob))
    return detjacob, invjacob  # eleNum x (3*3*8)


def dNdx_dNdy_dNdz_G(invjacob, dshapedr, dshapeds, dshapedt, eleNum):
    dshapedx = np.zeros((eleNum, 8 * 8), dtype=np.float32)
    dshapedy = np.zeros((eleNum, 8 * 8), dtype=np.float32)
    dshapedz = np.zeros((eleNum, 8 * 8), dtype=np.float32)
    for i in range(eleNum):
        for j in range(8):
            temp = invjacob[i, j * 9:(j + 1) * 9]
            invJ = np.reshape(temp, (3, 3))  # check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if i == 0 and j == 0:
            #     print(invJ)

            # dNdr = dshapedr[j,:]
            # dNds = dshapeds[j,:]
            # dNdt = dshapedt[j,:]
            dNdrst = np.zeros((3, 8), dtype=np.float32)

            dNdrst[0, :] = dshapedr[j, :]
            dNdrst[1, :] = dshapeds[j, :]
            dNdrst[2, :] = dshapedt[j, :]
            temp = invJ @ dNdrst;  # 3 x 8
            dNdx = temp[0, :]
            dNdy = temp[1, :]
            dNdz = temp[2, :]

            G1 = np.zeros((9, 24), dtype=np.float32)
            G1[0, ::3] = dNdx
            G1[1, ::3] = dNdx
            G1[2, ::3] = dNdx

            dshapedx[i, j * 8:(j + 1) * 8] = dNdx;
            dshapedy[i, j * 8:(j + 1) * 8] = dNdy;
            dshapedz[i, j * 8:(j + 1) * 8] = dNdz;

    return dshapedx, dshapedy, dshapedz  # ,G


def kinematicStiffnessLinear(dshapedx, dshapedy, dshapedz,eleNum):
    B0 = np.zeros((eleNum, 6 * 24 * 8), dtype=np.float32)
    for i in range(eleNum):
        for j in range(8):
            dNdx = dshapedx[i, j * 8:(j + 1) * 8]
            dNdy = dshapedy[i, j * 8:(j + 1) * 8]
            dNdz = dshapedz[i, j * 8:(j + 1) * 8]

            B0i = np.zeros((6, 24), dtype=np.float32)
            B0i[0, ::3] = dNdx;
            B0i[1, 1::3] = dNdy;
            B0i[2, 2::3] = dNdz;

            B0i[3, ::3] = dNdy;
            B0i[3, 1::3] = dNdx;

            B0i[4, 1::3] = dNdz;
            B0i[4, 2::3] = dNdy;

            B0i[5, ::3] = dNdz;
            B0i[5, 2::3] = dNdx;

            # if i == 0 and j == 0:
            #     print(B0i[0, :])
            B0[i, j * 6 * 24:(j + 1) * 6 * 24] = B0i.flatten()
    return B0


def shapeFunction():
    # https://www.mathworks.com/matlabcentral/fileexchange/13508-multi-dimensional-gauss-points-and-weights
    # https://www.mathworks.com/matlabcentral/fileexchange/6862-gauss3d
    gp = 0.577350269189626
    gausspoint = np.zeros((8, 3), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                gausspoint[i * 4 + j * 2 + k, :] = [(-1) ** i * gp, (-1) ** j * gp, (-1) ** k * gp]
    wt = np.ones((8, 1), dtype=np.float32)

    r = gausspoint[:, 0]
    s = gausspoint[:, 1]
    t = gausspoint[:, 2]
    # shape functions, [N1,N2,N3,N4;] for every gauss point
    shape = np.zeros((8, 8), dtype=np.float32)
    shape[:, 0] = (1 - r) * (1 - s) * (1 - t) / 8
    shape[:, 1] = (1 + r) * (1 - s) * (1 - t) / 8
    shape[:, 2] = (1 - r) * (1 + s) * (1 - t) / 8
    shape[:, 3] = (1 + r) * (1 + s) * (1 - t) / 8

    shape[:, 4] = (1 - r) * (1 - s) * (1 + t) / 8
    shape[:, 5] = (1 + r) * (1 - s) * (1 + t) / 8
    shape[:, 6] = (1 - r) * (1 + s) * (1 + t) / 8
    shape[:, 7] = (1 + r) * (1 + s) * (1 + t) / 8

    # derivatives, [dN1dxi,dN2dxi,dN3dxi,dN4dxi;] for every gauss point
    dshapedr = np.zeros((8, 8), dtype=np.float32)
    dshapedr[:, 0] = -(1 - s) * (1 - t) / 8
    dshapedr[:, 1] =  (1 - s) * (1 - t) / 8
    dshapedr[:, 2] = -(1 + s) * (1 - t) / 8
    dshapedr[:, 3] =  (1 + s) * (1 - t) / 8
    
    dshapedr[:, 4] = -(1 - s) * (1 + t) / 8
    dshapedr[:, 5] =  (1 - s) * (1 + t) / 8
    dshapedr[:, 6] = -(1 + s) * (1 + t) / 8
    dshapedr[:, 7] =  (1 + s) * (1 + t) / 8
   
    # [dN1deta,dN2deta,dN3deta,dN4deta;] for every gauss point
    dshapeds = np.zeros((8, 8), dtype=np.float32)
    dshapeds[:, 0] = -(1 - r) * (1 - t) / 8
    dshapeds[:, 1] = -(1 + r) * (1 - t) / 8
    dshapeds[:, 2] =  (1 - r) * (1 - t) / 8
    dshapeds[:, 3] =  (1 + r) * (1 - t) / 8
    
    dshapeds[:, 4] = -(1 - r) * (1 + t) / 8
    dshapeds[:, 5] = -(1 + r) * (1 + t) / 8
    dshapeds[:, 6] =  (1 - r) * (1 + t) / 8
    dshapeds[:, 7] =  (1 + r) * (1 + t) / 8

    dshapedt = np.zeros((8, 8), dtype=np.float32)
    dshapedt[:, 0] = -(1 - r) * (1 - s) / 8
    dshapedt[:, 1] = -(1 + r) * (1 - s) / 8
    dshapedt[:, 2] = -(1 - r) * (1 + s) / 8
    dshapedt[:, 3] = -(1 + r) * (1 + s) / 8
    
    dshapedt[:, 4] =  (1 - r) * (1 - s) / 8
    dshapedt[:, 5] =  (1 + r) * (1 - s) / 8
    dshapedt[:, 6] =  (1 - r) * (1 + s) / 8
    dshapedt[:, 7] =  (1 + r) * (1 + s) / 8
    return wt, shape, dshapedr, dshapeds, dshapedt

def ISOelasticitytensor(E = 1, nu = 0.33):
    D = np.zeros((6, 6), dtype=np.float32)
    D[0, 0] = 1 - nu
    D[1, 1] = 1 - nu
    D[2, 2] = 1 - nu

    D[0, 1] = nu
    D[0, 2] = nu
    D[1, 0] = nu
    D[1, 2] = nu
    D[2, 0] = nu
    D[2, 1] = nu

    D[3, 3] = (1 - 2 * nu) / 2
    D[4, 4] = (1 - 2 * nu) / 2
    D[5, 5] = (1 - 2 * nu) / 2

    D = D * E / ((1 + nu) * (1 - 2 * nu))
    return D
def LocalIntegratedMatrices(D,B0, detjacob, wt, eleNum):
    # print(D)
    # K = np.zeros((eleNum, 24, 24), dtype=np.float32)
    # F = np.zeros((eleNum, 24, 6), dtype=np.float32)
    for i in range(eleNum):
        Ke = np.zeros((24, 24), dtype=np.float32)
        Fe = np.zeros((24, 6), dtype=np.float32)
        intB = np.zeros((6,24), dtype=np.float32)
        for j in range(8):
            B0i = B0[i, j * 6 * 24:(j + 1) * 6 * 24]
            B0i = np.reshape(B0i, (6, 24))
            detJ = detjacob[i, j]
            B0iT = B0i.transpose()
            Ke += B0iT @ D @ B0i * detJ * wt[j]
            Fe += B0iT @ D * detJ * wt[j]
            intB += B0i * detJ * wt[j]
        # K[i, :, :] = Ke
    return Ke,Fe,intB
    
def LocalKeFe(resolution,D0):
    h=1.0/resolution
    nodes = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [1,1,0],
                      [0,0,1],
                      [1,0,1],
                      [0,1,1],
                      [1,1,1]])*h
    eles = np.array([[0,1,2,3,4,5,6,7]])
    eleNum=1
    # from elasticity3Dhex import Jacobian,dNdx_dNdy_dNdz_G,kinematicStiffnessLinear
    wt, shape, dshapedr, dshapeds, dshapedt = shapeFunction()
    detjacob, invjacob = Jacobian(nodes, eles, dshapedr, dshapeds, dshapedt, eleNum)
    dshapedx, dshapedy, dshapedz = dNdx_dNdy_dNdz_G(invjacob, dshapedr, dshapeds, dshapedt, eleNum)
    B0 = kinematicStiffnessLinear(dshapedx, dshapedy, dshapedz,eleNum)
    Ke,Fe,intB = LocalIntegratedMatrices(D0,B0, detjacob, wt, eleNum)
    return Ke,Fe,intB
if __name__ == "__main__":
    D0 = ISOelasticitytensor(E = 1e6, nu = 0.3)
    resolution=40
    h=1.0/resolution
    nodes = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [1,1,0],
                      [0,0,1],
                      [1,0,1],
                      [0,1,1],
                      [1,1,1]])*h
    eles = np.array([[0,1,2,3,4,5,6,7]])
    eleNum=1
    # from elasticity3Dhex import Jacobian,dNdx_dNdy_dNdz_G,kinematicStiffnessLinear
    wt, shape, dshapedr, dshapeds, dshapedt = shapeFunction()
    detjacob, invjacob = Jacobian(nodes, eles, dshapedr, dshapeds, dshapedt, eleNum)
    dshapedx, dshapedy, dshapedz = dNdx_dNdy_dNdz_G(invjacob, dshapedr, dshapeds, dshapedt, eleNum)
    B0 = kinematicStiffnessLinear(dshapedx, dshapedy, dshapedz,eleNum)
    Ke,Fe,intB = LocalIntegratedMatrices(D0,B0, detjacob, wt, eleNum)
