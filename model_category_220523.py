import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self,NUM_CATGR):
        super(Classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, (4,4), stride=(2,2), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, (4,4), stride=(2,2), padding=(1,1)),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, (4,4), stride=(2,2), padding=(1,1)),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, (4,4), padding=(1,1)),
            #nn.InstanceNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 256, (4,4), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 128, (4, 4), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, (4, 4), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc = nn.Linear( 28 * 28, NUM_CATGR)
        self.actvsig = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        #x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        x = self.fc( x )
        return self.actvsig(x)

BATCH_SIZE = 32


if __name__ == '__main__':
    FakeIm = torch.randn((BATCH_SIZE,3,256,256)).cuda()
    D = Classifier(11).cuda()
    print( D(FakeIm).size() )
