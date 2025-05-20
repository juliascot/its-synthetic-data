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



if __name__ == "__main__":
    augmented_tensor = create_dense_tensor(filename, rank)




