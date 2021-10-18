import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # encoding layers
        self.conv1 = nn.Conv1d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv1d(32, 64, 3, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, stride=1)
        self.conv5 = nn.Conv1d(128, 256, 3, stride=1)
        self.conv6 = nn.Conv1d(256, 512, 3, stride=1)
        

        # decoding layers
        self.deconv6 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1)
        self.deconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1)
        self.deconv1 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1)
        
        # activations
        self.activ1 = nn.PReLU() #not inplace, I want to copy
        self.activ2 = nn.Tanh()
        

    def forward(self, x):
        
        # encoder
        x = self.conv1(x)
        res1 = self.activ1(x)
        
        x = self.conv2(res1)
        res2 = self.activ1(x)
        
        x = self.conv3(res2)
        res3 = self.activ1(x)
        
        x = self.conv4(res3)
        res4 = self.activ1(x)
        
        x = self.conv5(res4)
        res5 = self.activ1(x)
        
        x = self.conv6(res5)
        x = self.activ1(x)   
        
        
        
        # decoder
        x = self.deconv6(x)
        x = self.activ1(x)
        x += res5
        
        x = self.deconv5(x)
        x = self.activ1(x)
        x += res4
        
        x = self.deconv4(x)
        x = self.activ1(x)
        x += res3
        
        x = self.deconv3(x)
        x = self.activ1(x)
        x += res2
        
        x = self.deconv2(x)
        x = self.activ1(x)
        x += res1
        
        x = self.deconv1(x)
        x = self.activ2(x)
        
        
        return x
        
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # encoding layers
        self.encode = nn.Sequential(
            nn.Conv1d(2, 16, 3, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, 3, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),            
            nn.Conv1d(32, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),            
            nn.Conv1d(64, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, 3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),            
           
        )
        
        self.embed = nn.Sequential(
            nn.Conv1d(256, 512, 3, stride=1),
            nn.LeakyReLU(0.1), 
        )
        
        self.head = nn.Sequential(
            nn.Linear(_,_),
            nn.ReLU(),
            nn.Linear(_,_),
            nn.ReLU(),
            nn.Linear(_,1),
            nn.Sigmoid()
        )
        
    def forward(self, X, Xc):
        # concatenate both ground truth or predicted + original signal (used as conditional)
        b, c, l = X.shape
        Xi = torch.cat(X,Xc,axis=1)
        
        Xi = self.encode(Xi)
        Xi = self.embed(Xi)
        
        Xi = Xi.reshape(b,-1)
        proba = self.head(Xi)
        
        return proba
        

            