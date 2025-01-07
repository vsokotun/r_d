import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        
        self.initial = nn.Linear(latent_dim, 512 * 3 * 4)
        
        self.main = nn.Sequential(
            # input: 512 x 3 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state: 256 x 6 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state: 128 x 12 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state: 64 x 24 x 32

            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # final state: channels x 48 x 64
        )

    def forward(self, x):
        x = self.initial(x)
        x = x.view(x.size(0), 512, 3, 4)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # input: channels x 48 x 64
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: 64 x 24 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state: 128 x 12 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state: 256 x 6 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state: 512 x 3 x 4

            nn.Conv2d(512, 1, kernel_size=(3, 4), stride=1, padding=0, bias=False)
            # final state: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1) 
