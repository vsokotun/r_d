import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

latent_dim = 100
num_classes = 10
img_w = 28
img_h = 28
channels = 1
img_shape = (channels, img_w, img_h)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 128
epochs = 200
lr = 0.0001

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_loader = DataLoader(datasets.MNIST(".", train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)

os.makedirs('generated_images', exist_ok=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, int(torch.prod(torch.tensor(img_shape))))

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_input = self.label_embedding(labels)
        d_input = torch.cat((img_flat, label_input), -1)
        validity = self.model(d_input)
        return validity

generator = Generator(latent_dim, num_classes, img_shape).to(device)
discriminator = Discriminator(num_classes, img_shape).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        optimizer_G.zero_grad()
        noise = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),), device=device)
        gen_imgs = generator(noise, gen_labels)

        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def generate_images(generator, latent_dim, n_row=10):
        generator.eval()
        noise = torch.randn(n_row, latent_dim, device=device)
        labels = torch.arange(0, n_row, device=device)
        with torch.no_grad():
            gen_imgs = generator(noise, labels)

        gen_imgs = gen_imgs.cpu().numpy()
        fig, axs = plt.subplots(1, n_row, figsize=(10, 2))
        for i in range(n_row):
            axs[i].imshow(gen_imgs[i, 0, :, :], cmap="gray")
            axs[i].axis('off')
            axs[i].set_title(f"{labels[i].item()}")
        save_path = os.path.join('generated_images', f'epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close(fig)

    generate_images(generator, latent_dim)
    generator.train()


torch.save(generator.state_dict(), 'generator_cgan.pth')
