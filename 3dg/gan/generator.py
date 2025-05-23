import torch.nn as nn
from helper_funcs import SpecialSigmoid

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 5 * 21 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 5, 21)),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=(1,0)),
            SpecialSigmoid()
        )

    def forward(self, x):
        return self.main(x)