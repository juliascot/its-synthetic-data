import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import keras
from typing import Any
from collections.abc import Callable
from generator import Generator
from discriminator import Discriminator
from helper_funcs import *




################################
#       HYPERPARAMETERS        #
################################


filename = "Getting_Started_Reordered.csv"
rank = 5
pre_reconstruction_augmentation_values = [-2, -1, 1, 2] # Offset first correct attempt on every question by this amount
post_reconstruction_augmentation_values = [0.9, 1.1] # Multiply the whole reconstructed tensor by these amounts to increase its size
l2 = 0
epochs = 1000
noise_dimension = 100
batch_size = 30
gp_weight = 10.0
critic_iters_per_gen = 5

generator_optimizer = keras.optimizers.Adam()
discriminator_optimizer = keras.optimizers.Adam()

def critic_loss(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    return -(torch.mean(real_scores) - torch.mean(fake_scores))

def generator_loss(fake_scores: np.ndarray) -> float:
    return -torch.mean(fake_scores)




################################
#             GAN              #
################################


class WGAN():
    def __init__(self, generator: Generator, discriminator: Discriminator, noise_dimension: int, device: torch.device, gp_weight: float = 10, critic_iters_per_gen: int = 5):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dimension = noise_dimension
        self.device = device
        self.gp_weight = gp_weight
        self.critic_iters = critic_iters_per_gen
         
    def gradient_penalty(self, batch_size: int, real_slices: np.ndarray, fake_slices: np.ndarray) -> float:
        alpha = torch.rand(batch_size, 1, 1, 1)
        interpolated = real_slices + alpha * (fake_slices - real_slices)
        interpolated.requires_grad_(True)

        mixed_scores = self.discriminator(interpolated)

        gradients = torch.autograd.grad(mixed_scores, interpolated, grad_outputs=torch.ones_like(mixed_scores))[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gp = self.gp_weight * ((gradient_norm - 1) ** 2).mean()
        return gp
    
    def compile(self, d_optimizer: Any, g_optimizer: Any, d_loss_fn: Callable[[np.ndarray, np.ndarray], float], g_loss_fn: Callable[[np.ndarray], float]) -> None:
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, real_slices: np.ndarray):

        batch_size = real_slices.size(0)

        for _ in range(self.critic_iters):
            noise = torch.randn(batch_size, self.noise_dimension).to(self.device)
            fake_slices = self.generator(noise).detach()

            real_scores = self.discriminator(real_slices)
            fake_scores = self.discriminator(fake_slices)

            discriminator_loss = critic_loss(real_scores, fake_scores) + self.gradient_penalty(batch_size, real_slices, fake_slices)

            self.d_optimizer.zero_grad()
            discriminator_loss.backward()
            self.d_optimizer.step()

        noise = torch.randn(real_slices.size(0), self.noise_dimension).to(self.device)
        fake_slices = self.generator(noise)
        fake_scores = self.discriminator(fake_slices)

        gen_loss = generator_loss(fake_scores)

        self.g_optimizer.zero_grad()
        gen_loss.backward()
        self.g_optimizer.step()

        return discriminator_loss.item(), gen_loss.item()




################################
#             MAIN             #
################################


if __name__ == "__main__":

    augmented_tensor = create_dense_tensor(filename, rank, pre_reconstruction_augmentation_values, post_reconstruction_augmentation_values, l2=l2)
    train_loader = torch.utils.data.DataLoader(augmented_tensor, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(noise_dimension).to(device)
    discriminator = Discriminator().to(device)

    wgan = WGAN(generator, discriminator, noise_dimension, device, gp_weight=gp_weight, critic_iters_per_gen=critic_iters_per_gen)
    wgan.compile(discriminator_optimizer, generator_optimizer, critic_loss, generator_loss)


    for i in range(epochs):
        for i, batch_slices in enumerate(train_loader):
            discriminator_loss_output, generator_loss_output = wgan.train_step(batch_slices)

    # loop through epochs






