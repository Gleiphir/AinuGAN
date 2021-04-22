# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F

#from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4


class ResBlk(nn.Module):
    def __init__(self, layers,channel,stride=1):
        super(ResBlk, self).__init__()
        assert len(channel) ==2
        self.layers = nn.Sequential(layers)
        if stride ==-2:
            self.bypass = nn.Sequential(
                nn.Conv2d(channel[0],channel[1],1,1,padding=0),
                nn.Upsample(scale_factor=2)
                )
        elif stride ==2:
            self.bypass = nn.Sequential(
                nn.Conv2d(channel[0],channel[1],1,1,padding=0),
                nn.AvgPool2d(2)
                )
        else:
            self.bypass = nn.Conv2d(channel[0],channel[1],1,1,padding=0)
        
         
    def forward(self, x):
        print(x.size(),self.layers(x).size() , self.bypass(x).size() )
        return self.layers(x) + self.bypass(x)




class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        #
        self.ln = nn.Linear(z_dim, 1024 * w_g * w_g)
        #
        self.model = nn.Sequential(

            nn.ConvTranspose2d(1024, 512, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, channels, (3,3), stride=(1,1), padding=(1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(self.ln(z).view(-1, 1024, w_g, w_g))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq = nn.Sequential(
		    nn.Conv2d(channels, 128, (3,3), stride=(1,1), padding=(1,1)),
		    nn.LeakyReLU(leak),
		    nn.Conv2d(128, 128, (4,4), stride=(2,2), padding=(1,1)),
		    nn.LeakyReLU(leak),
		    nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(1,1)),
		    nn.LeakyReLU(leak),
		    nn.Conv2d(256, 256, (4,4), stride=(2,2), padding=(1,1)),
		    nn.LeakyReLU(leak),
		    nn.Conv2d(256, 512, (3,3), stride=(1,1), padding=(1,1)),
		    nn.LeakyReLU(leak),
	    )
        self.fc = nn.Linear(w_g * w_g * 512, 1)

    def forward(self, x):
        m = x
        m = self.seq(m)
        return self.fc(m.view(-1,w_g * w_g * 512))

