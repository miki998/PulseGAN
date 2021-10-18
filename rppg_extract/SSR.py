"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

import numpy as np

def build_p(τ, k, l, U, Λ):
    """
    builds P
    Parameters
    ----------
    k: int
        The frame index
    l: int
        The temporal stride to use
    U: numpy.ndarray
        The eigenvectors of the c matrix (for all frames up to counter).
    Λ: numpy.ndarray
        The eigenvalues of the c matrix (for all frames up to counter).
    Returns
    -------
    p: numpy.ndarray
        The p signal to add to the pulse.
    """
    # SR'
    SR = np.zeros((3, l), 'float64')  # dim: 3xl
    z = 0

    for t in range(τ, k, 1):  # 6, 7
        a = Λ[0, t]
        b = Λ[1, τ]
        c = Λ[2, τ]
        d = U[:, 0, t].T
        e = U[:, 1, τ]
        f = U[:, 2, τ]
        g = U[:, 1, τ].T
        h = U[:, 2, τ].T
        x1 = a / b
        x2 = a / c
        x3 = np.outer(e, g)
        x4 = np.dot(d, x3)
        x5 = np.outer(f, h)
        x6 = np.dot(d, x5)
        x7 = np.sqrt(x1)
        x8 = np.sqrt(x2)
        x9 = x7 * x4
        x10 = x8 * x6
        x11 = x9 + x10
        SR[:, z] = x11  # 8 | dim: 3
        z += 1

    # build p and add it to the final pulse signal
    s0 = SR[0, :]  # dim: l
    s1 = SR[1, :]  # dim: l
    p = s0 - ((np.std(s0) / np.std(s1)) * s1)  # 10 | dim: l
    p = p - np.mean(p)  # 11
    return p  # dim: l

def build_correlation_matrix(V):
    # V dim: (W×H)x3
    #V = np.unique(V, axis=0)
    V_T = V.T  # dim: 3x(W×H)
    N = V.shape[0]
    # build the correlation matrix
    C = np.dot(V_T, V)  # dim: 3x3
    C = C / N

    return C

def eigs(C):
    """
    get eigenvalues and eigenvectors, sort them.
    Parameters
    ----------
    C: numpy.ndarray
        The RGB values of skin-colored pixels.
    Returns
    -------
    Λ: numpy.ndarray
        The eigenvalues of the correlation matrix
    U: numpy.ndarray
        The (sorted) eigenvectors of the correlation matrix
    """
    # get eigenvectors and sort them according to eigenvalues (largest first)
    L, U = np.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3
    idx = L.argsort()  # dim: 3x1
    idx = idx[::-1]  # dim: 1x3
    L_ = L[idx]  # dim: 3
    U_ = U[:, idx]  # dim: 3x3

    return L_, U_
    
def SSR(X,fps,windows=1.6):
    """
    desc: from roi obtain a one time signal POS
    
    args:
        - roi::[array<array<array<float> > >]
            stream of images
    ret: 
        - P::[array<float>] 
            1d signal
    
    """
    
    K = len(X)
    l = fps

    P = np.zeros(K)  # 1 | dim: K
    # store the eigenvalues Λ and the eigenvectors U at each frame
    L = np.zeros((3, K), dtype='float64')  # dim: 3xK
    U = np.zeros((3, 3, K), dtype='float64')  # dim: 3x3xK

    for k in range(K):
        n_roi = len(X[k])
        VV = []

        for r in range(n_roi):
            V = X[k][r].astype(np.float64)
            idx = V!=0
            idx2 = np.logical_and(np.logical_and(idx[:,0], idx[:,1]), idx[:,2])
            V_skin_only = V[idx2]
            VV.append(V_skin_only)

        VV = np.vstack(VV)

        C = build_correlation_matrix(VV)  #dim: 3x3

        # get: eigenvalues Λ, eigenvectors U
        L[:,k], U[:,:,k] = eigs(C)  # dim Λ: 3 | dim U: 3x3

        # build p and add it to the pulse signal P
        if k >= l:  # 5
            tau = k - l  # 5
            p = build_p(tau, k, l, U, L)  # 6, 7, 8, 9, 10, 11 | dim: l
            P[tau:k] += p  # 11

        if np.isnan(np.sum(P)):
            print('NAN')
            print(X[k])

    bvp = P

    return bvp
