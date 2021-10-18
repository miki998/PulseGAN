"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

import numpy as np

def CHROM(X):
    """
    desc: from RGB spatial-average obtain a one time signal POS
    
    args:
        - X::[array<array<float> >]
            RGB spatial-averaged array

    ret: 
        - C::[array<float>] 
            1d signal
    
    """
    
    Xs = X[0]-X[1]
    Ys = X[0]+X[1]-2*X[2]
    
    alpha = np.std(Xs)/np.std(Ys)
    
    C = Xs-alpha*Ys
    return C