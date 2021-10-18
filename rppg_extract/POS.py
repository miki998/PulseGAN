"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

import numpy as np

def POS(X,fps,windows=1.6):
    """
    desc: from RGB spatial-average obtain a one time signal POS
    
    args:
        - X::[array<array<float> > ]
            RGB spatial-averaged array
    ret: 
        - h::[array<float>] 
            1d signal
    
    """
    P = np.array([[0,1,-1],[-2,1,1]])
    wlen = int(windows * fps)


    N = X.shape[1]
    
    # Initialize (1)
    h = np.zeros(N)
    for n in range(N):
        # Start index of sliding window (4)
        m = n - wlen + 1

        if m >= 0:

            # Temporal normalization (5)
            cn = X[:,m:(n+1)]
            mu = np.mean(cn,axis=1)
            cn[0] = cn[0] / mu[0]
            cn[1] = cn[1] / mu[1]
            cn[2] = cn[2] / mu[2]
            
            # Projection (6)

            s = np.matmul(P,cn)
            # Tuning (7)
            hn = np.add(s[0, :], np.std(s[0, :])/np.std(s[1, :])*s[1, :])
            # Overlap-adding (8)
            h[m:(n+1)] = np.add(h[m:(n+1)], hn - np.mean(hn))
            
    return h
    