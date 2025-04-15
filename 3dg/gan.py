import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorly.decomposition import parafac
from tensor import Tensor

filename = "Getting_Started_Processed.csv"
rank = 8
l2 = 0


def special_sigmoid(input: any) -> any:
    return 1 / (1 + np.exp(-6 * input + 3))


def special_sigmoid_inverse(input: any) -> any:
    return (3 - np.log(1 / input - 1)) / 6


def create_dense_tensor(filename: str, rank: int, l2: float) -> np.ndarray:
    initial_tensor = Tensor(filename, is_student_outside=True, is_augmented=True)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    augmented_tensor = np.copy(reconstructed_tensor)
    augmented_tensor = np.vstack((augmented_tensor, augmented_tensor * 0.95, augmented_tensor * 1.05))

    return special_sigmoid(augmented_tensor)


# Assumes all slices are of shape (dim1, dim2)
class Generator(nn.Module):
    def __init__(self, noise_dim, slice_shape):
        self.slice_shape = slice_shape
        super().__init__()
        self.output_dim = slice_shape[0] * slice_shape[1]
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, *self.slice_shape)


class Discriminator(nn.Module):
    def __init__(self, slice_shape):
        super().__init__()
        input_dim = slice_shape[0] * slice_shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def train_gan(slices, noise_dim=100, epochs=1000, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_shape = slices.shape[1:]
    generator = Generator(noise_dim, slice_shape).to(device)
    discriminator = Discriminator(slice_shape).to(device)

    criterion = nn.BCELoss()
    optim_G = optim.Adam(generator.parameters(), lr=0.0002)
    optim_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    data = torch.tensor(slices, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        idx = torch.randint(0, data.size(0), (batch_size,))
        real = data[idx]

        # Labels
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # --- Train Discriminator ---
        z = torch.randn((batch_size, noise_dim), device=device)
        fake = generator(z)

        out_real = discriminator(real)
        out_fake = discriminator(fake.detach())

        loss_D = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # --- Train Generator ---
        out_fake = discriminator(fake)
        loss_G = criterion(out_fake, real_labels)  # Fool the discriminator
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    return generator


def generate_slices(generator, num_slices, noise_dim=100):
    z = torch.randn((num_slices, noise_dim))
    with torch.no_grad():
        fake_slices = generator(z).numpy()
    return fake_slices


def graph_student_slices(slices: np.ndarray) -> None:
    # Extract prior knowledge (a) and acquired knowledge (b)
    all_extracted_info = []

    for student_num, student_matrix in enumerate(slices):

        extracted_info_a = []
        extracted_info_b = []

        for question in student_matrix:

            X = np.arange(1, len(question) + 1)

            popt, pcov = curve_fit(power_law, X, question, p0=[1, 1], bounds=([0, 0], [1, 1]))

            extracted_info_a.append(popt[0])
            extracted_info_b.append(popt[1])

        
        all_extracted_info.append([extracted_info_a, extracted_info_b])


    # Graph the extracted values
    plt.figure()

    for student_num in range(len(slices)):
        plt.scatter(all_extracted_info[student_num-1][0], all_extracted_info[student_num-1][1])

    plt.title('Across all questions',fontsize=8)
    plt.suptitle(f'Students\' Learning Curves',fontsize=16, y=0.97)
    plt.xlabel("$\t{a}$: prior knowledge")
    plt.ylabel("$\t{b}$: learning rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    augmented_tensor = create_dense_tensor(filename, rank, l2)
    np.random.shuffle(augmented_tensor)

    generator = train_gan(augmented_tensor)

    synthetic_slices = generate_slices(generator, 1)

    print(f"Synthetic: {special_sigmoid_inverse(synthetic_slices)}")
    print(f"Real slice: {special_sigmoid_inverse(augmented_tensor[:1][:1])}")

