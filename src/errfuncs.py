import torch
import torch.nn as nn





class Gloss(nn.Module):  
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    
    def forward(self, rppg_gt, rppg_pred, proba_pred, lbd=.1, beta=.2):
        # accumulating loss for all sample
        loss = 0        
        b, c, l = rppg_gt.shape

        for i in range(b):
            d = proba_pred[i]
            Xc = rppg_gt[i]
            Xg = rppg_pred[i]
            
            loss += 1/2 * (d - 1)**2
            loss += lbd * np.sum(np.abs(Xc-Xg))
            loss += beta * np.sum(np.abs(fft(Xc) - fft(Xg)))


        return loss
    
    
    def fft(self,Xc,Xg):
        return

class Dloss(nn.Module):  
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    
    def forward(self, proba_fake, proba_real):      
        # accumulating loss for all sample        
        b, c, l = proba_fake.shape
        loss = 0
        
        for i in range(b):
            df = proba_fake[i]
            dr = proba_real[i]
            
            loss += 1/2 * df ** 2 + 1/2 * (dr-1)**2

        return loss