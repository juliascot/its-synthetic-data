import torch
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
    def __init__(self, generator: Generator, discriminator: Discriminator, noise_dimension: int, discriminator_extra_iters: int, gp_weight: float = 10, critic_iters_per_gen: int = 5):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dimension = noise_dimension
        self.discriminator_extra_iters = discriminator_extra_iters
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

    def train_step(self, real_slices: np.ndarray) -> dict:
        for _ in range(self.critic_iters):
            pass
        # loop through the critic_iters to generate fake slices and train discriminator (calculate loss, update weights)
        # generate another image and train generator (calculate loss, update weights)
        # return losses
        return








################################
#             MAIN             #
################################


if __name__ == "__main__":
    augmented_tensor = create_dense_tensor(filename, rank, pre_reconstruction_augmentation_values, post_reconstruction_augmentation_values, l2=l2)

    # start generator and discriminator
    # start WGAN, passing through all of the hyperparameters
    # loop through epochs






