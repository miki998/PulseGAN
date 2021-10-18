"""

@author: Chun Hei Michael Chan
@copyright: Copyright Logitech
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: cchan5@logitech.com

"""

def SCch(X,channel=1):
    """
    desc: select a channel from image and return it
    
    args: 
        - X::[array<array<int>>]
            image
    ret: 
        - monochannel::[array<array<int>>]
            image with only one channel
    
    """
    monochannel = X[channel,:]
    return monochannel
