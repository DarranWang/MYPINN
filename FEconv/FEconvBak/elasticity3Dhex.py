import torch
# import torch.autograd as ag
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np

# loss = Loss3DgeneralMesh(data, ref, device, K, b, free, edofMat)


def Loss3DgeneralMesh(input, output, device, K, b, free, edofMat):
    # eleNum = 792
    nodNum = 1035
    # input [bs,eleNum]
    # K [eleNum,24,24]

    KK = K.repeat((input.shape[0], 1, 1, 1))
    # print(input.shape)
    rho = input.repeat((24, 24, 1, 1)).permute([2, 3, 0, 1])
    # rho*K: [bs,eleNum,24,24]
    KK = KK * rho
    # print(KK[idata,iele]-K[iele]*input[idata,iele])
    u = torch.zeros((input.shape[0], nodNum * 3)).to(device)
    u[:, free] = output  # [bs,dofs]

    # [bs,eleNum,24,1]
    U = torch.zeros((input.shape[0], input.shape[1], 24, 1)).to(device)
    U[:, :, :, 0] = u[:, edofMat]
    # U[id,iele,:,0] = u[id,edofMat[iele,:]]
    UT = U.permute([0, 1, 3, 2])

    losst1 = torch.matmul(torch.matmul(UT, KK), U).sum()

    # b = torch.from_numpy(b)
    bb = b.repeat((input.shape[0], 1, 1))
    bb = bb.squeeze(2)

    FU = (output * bb).sum()

    #     print(losst1)
    #     print(FU)
    return losst1 / 2 - FU


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
    shape[:, 0] = (1 + r) * (1 + s) * (1 - t) / 8
    shape[:, 1] = (1 - r) * (1 + s) * (1 - t) / 8
    shape[:, 2] = (1 - r) * (1 - s) * (1 - t) / 8
    shape[:, 3] = (1 + r) * (1 - s) * (1 - t) / 8

    shape[:, 4] = (1 + r) * (1 + s) * (1 + t) / 8
    shape[:, 5] = (1 - r) * (1 + s) * (1 + t) / 8
    shape[:, 6] = (1 - r) * (1 - s) * (1 + t) / 8
    shape[:, 7] = (1 + r) * (1 - s) * (1 + t) / 8

    # derivatives, [dN1dxi,dN2dxi,dN3dxi,dN4dxi;] for every gauss point
    dshapedr = np.zeros((8, 8), dtype=np.float32)
    dshapedr[:, 0] = (1 + s) * (1 - t) / 8
    dshapedr[:, 1] = -(1 + s) * (1 - t) / 8
    dshapedr[:, 2] = -(1 - s) * (1 - t) / 8
    dshapedr[:, 3] = (1 - s) * (1 - t) / 8

    dshapedr[:, 4] = (1 + s) * (1 + t) / 8
    dshapedr[:, 5] = -(1 + s) * (1 + t) / 8
    dshapedr[:, 6] = -(1 - s) * (1 + t) / 8
    dshapedr[:, 7] = (1 - s) * (1 + t) / 8
    # [dN1deta,dN2deta,dN3deta,dN4deta;] for every gauss point
    dshapeds = np.zeros((8, 8), dtype=np.float32)
    dshapeds[:, 0] = (1 + r) * (1 - t) / 8
    dshapeds[:, 1] = (1 - r) * (1 - t) / 8
    dshapeds[:, 2] = -(1 - r) * (1 - t) / 8
    dshapeds[:, 3] = -(1 + r) * (1 - t) / 8

    dshapeds[:, 4] = (1 + r) * (1 + t) / 8
    dshapeds[:, 5] = (1 - r) * (1 + t) / 8
    dshapeds[:, 6] = -(1 - r) * (1 + t) / 8
    dshapeds[:, 7] = -(1 + r) * (1 + t) / 8

    dshapedt = np.zeros((8, 8), dtype=np.float32)
    dshapedt[:, 0] = -(1 + r) * (1 + s) / 8
    dshapedt[:, 1] = -(1 - r) * (1 + s) / 8
    dshapedt[:, 2] = -(1 - r) * (1 - s) / 8
    dshapedt[:, 3] = -(1 + r) * (1 - s) / 8

    dshapedt[:, 4] = (1 + r) * (1 + s) / 8
    dshapedt[:, 5] = (1 - r) * (1 + s) / 8
    dshapedt[:, 6] = (1 - r) * (1 - s) / 8
    dshapedt[:, 7] = (1 + r) * (1 - s) / 8
    return wt, shape, dshapedr, dshapeds, dshapedt


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


def LocalStiffnessMatrix(B0, detjacob, wt, eleNum):
    E = 1
    nu = 0.3
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
    # print(D)
    K = np.zeros((eleNum, 24, 24), dtype=np.float32)
    for i in range(eleNum):
        Ke = np.zeros((24, 24), dtype=np.float32)
        for j in range(8):
            B0i = B0[i, j * 6 * 24:(j + 1) * 6 * 24]
            B0i = np.reshape(B0i, (6, 24))
            detJ = detjacob[i, j]
            B0iT = B0i.transpose()
            Ke += B0iT @ D @ B0i * detJ * wt[j]
        K[i, :, :] = Ke
    return K


def LocalLoad(eleNum, eles, nodes, NeumannBC):
    gp = 0.577350269189626
    gausspoint = np.zeros((4, 3), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            gausspoint[i * 2 + j, :] = [(-1) ** i * gp, (-1) ** j * gp, 1]
    wt = np.ones((4, 1), dtype=np.float32)

    r = gausspoint[:, 0]
    s = gausspoint[:, 1]
    t = gausspoint[:, 2]
    # shape functions, [N1,N2,N3,N4;] for every gauss point
    shape = np.zeros((4, 8), dtype=np.float32)
    shape[:, 0] = (1 + r) * (1 + s) * (1 - t) / 8
    shape[:, 1] = (1 - r) * (1 + s) * (1 - t) / 8
    shape[:, 2] = (1 - r) * (1 - s) * (1 - t) / 8
    shape[:, 3] = (1 + r) * (1 - s) * (1 - t) / 8

    shape[:, 4] = (1 + r) * (1 + s) * (1 + t) / 8
    shape[:, 5] = (1 - r) * (1 + s) * (1 + t) / 8
    shape[:, 6] = (1 - r) * (1 - s) * (1 + t) / 8
    shape[:, 7] = (1 + r) * (1 - s) * (1 + t) / 8

    # derivatives, [dN1dxi,dN2dxi,dN3dxi,dN4dxi;] for every gauss point
    dshapedr = np.zeros((4, 8), dtype=np.float32)
    dshapedr[:, 0] = (1 + s) * (1 - t) / 8
    dshapedr[:, 1] = -(1 + s) * (1 - t) / 8
    dshapedr[:, 2] = -(1 - s) * (1 - t) / 8
    dshapedr[:, 3] = (1 - s) * (1 - t) / 8

    dshapedr[:, 4] = (1 + s) * (1 + t) / 8
    dshapedr[:, 5] = -(1 + s) * (1 + t) / 8
    dshapedr[:, 6] = -(1 - s) * (1 + t) / 8
    dshapedr[:, 7] = (1 - s) * (1 + t) / 8
    # [dN1deta,dN2deta,dN3deta,dN4deta;] for every gauss point
    dshapeds = np.zeros((4, 8), dtype=np.float32)
    dshapeds[:, 0] = (1 + r) * (1 - t) / 8
    dshapeds[:, 1] = (1 - r) * (1 - t) / 8
    dshapeds[:, 2] = -(1 - r) * (1 - t) / 8
    dshapeds[:, 3] = -(1 + r) * (1 - t) / 8

    dshapeds[:, 4] = (1 + r) * (1 + t) / 8
    dshapeds[:, 5] = (1 - r) * (1 + t) / 8
    dshapeds[:, 6] = -(1 - r) * (1 + t) / 8
    dshapeds[:, 7] = -(1 + r) * (1 + t) / 8

    dshapedt = np.zeros((4, 8), dtype=np.float32)
    dshapedt[:, 0] = -(1 + r) * (1 + s) / 8
    dshapedt[:, 1] = -(1 - r) * (1 + s) / 8
    dshapedt[:, 2] = -(1 - r) * (1 - s) / 8
    dshapedt[:, 3] = -(1 + r) * (1 - s) / 8

    dshapedt[:, 4] = (1 + r) * (1 + s) / 8
    dshapedt[:, 5] = (1 - r) * (1 + s) / 8
    dshapedt[:, 6] = (1 - r) * (1 - s) / 8
    dshapedt[:, 7] = (1 + r) * (1 - s) / 8

    R = np.zeros((eleNum, 24, 1), dtype=np.float32)

    for i in range(eleNum):
        Rl = np.zeros((24, 1), dtype=np.float32)
        if NeumannBC[i, 5] == 1:
            node = eles[i, :]
            coord = nodes[node, :]  # 8 x 3
            for j in range(4):
                N = np.zeros((3, 24), dtype=np.float32)
                N[0, ::3] = shape[j, :]
                N[1, 1::3] = shape[j, :]
                N[2, 2::3] = shape[j, :]

                dN = np.zeros((3, 8), dtype=np.float32)

                dN[0, :] = dshapedr[j, :]
                dN[1, :] = dshapeds[j, :]
                dN[2, :] = dshapedt[j, :]
                J = dN @ coord
                ss = J[0, :]
                tt = J[1, :]
                X1 = ss[0];
                Y1 = ss[1];
                Z1 = ss[2];
                X2 = tt[0];
                Y2 = tt[1];
                Z2 = tt[2];
                nn = np.array([Y1 * Z2 - Y2 * Z1, Z1 * X2 - Z2 * X1, X1 * Y2 - X2 * Y1])
                nn_norm = (nn[0] ** 2 + nn[1] ** 2 + nn[2] ** 2) ** 0.5;
                f = np.array([[0], [0], [1]])
                Rl += N.transpose() @ f * nn_norm;
        R[i, :, :] = Rl
    return R


def dofMat(eleNum, eles):
    edofMat = np.zeros((eleNum, 24), dtype=int)
    for i in range(eleNum):
        ele = eles[i, :]
        edofMat[i, ::3] = ele * 3
        edofMat[i, 1::3] = ele * 3 + 1
        edofMat[i, 2::3] = ele * 3 + 2
    return edofMat


def freedof(nodNum, fixednodes):
    fixnodeNum = fixednodes.shape[0]
    fixeddofs = np.zeros((fixnodeNum * 3, 1), dtype=int)
    fixednodes = fixednodes - 1
    fixeddofs[::3, 0] = fixednodes * 3
    fixeddofs[1::3, 0] = fixednodes * 3 + 1
    fixeddofs[2::3, 0] = fixednodes * 3 + 2
    dofs = np.arange(3 * nodNum)
    free = np.setdiff1d(dofs, fixeddofs)
    return free


def GlobalLoad_free(eleNum,nodNum, R, edofMat, free):
    b = np.zeros((nodNum * 3, 1), dtype=np.float32)
    for i in range(eleNum):
        dof = edofMat[i, :].tolist()
        b[dof, 0] += R[i, :, 0]
    return b[free]
