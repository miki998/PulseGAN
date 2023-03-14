def SCch(X,channel=1):
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
    monochannel::[array<float>] 
        1d signal
    """

    monochannel = X[channel,:]
    return monochannel
