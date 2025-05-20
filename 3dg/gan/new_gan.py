import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from generator import Generator
from discriminator import Discriminator
from helper_funcs import *


filename = "Getting_Started_Reordered.csv"
rank = 5
pre_reconstruction_augmentation_values = [-2, -1, 1, 2] # Offset first correct attempt on every question by this amount
post_reconstruction_augmentation_values = [0.9, 1.1] # Multiply the whole reconstructed tensor by these amounts to increase its size
l2 = 0
epochs = 1000
noise_dimension = 100
batch_size = 30



def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = torch.rand(batch_size, 1, 1, 1)
        interpolated = real_images + alpha * (fake_images - real_images)
        interpolated.requires_grad_(True)

        mixed_scores = discriminator(interpolated)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


if __name__ == "__main__":
    augmented_tensor = create_dense_tensor(filename, rank, pre_reconstruction_augmentation_values, post_reconstruction_augmentation_values)






