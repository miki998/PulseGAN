import numpy as np

def CHROM(X):
    """
    Information:
    ------------
    From RGB spatial-average obtain a one time signal POS

    Parameters
    ----------
    X      ::[2darray<float>]
        RGB spatial-averaged array
    channel::[int]

    Returns
    -------
    C::[array<float>] 
        1d signal
    """
    
    Xs = X[0]-X[1]
    Ys = X[0]+X[1]-2*X[2]
    
    alpha = np.std(Xs)/np.std(Ys)
    
    C = Xs-alpha*Ys
    return C