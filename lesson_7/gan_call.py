import torch
import torch.nn as nn
import matplotlib.pyplot as plt


latent_dim = 100
num_classes = 10
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


generator = Generator(latent_dim, num_classes, img_shape).to(device)

generator.load_state_dict(torch.load("cifar_generator_cgan.pth", weights_only=True, map_location=device))
generator.eval()


def generate_specific_digit(generator, digit, latent_dim):
    noise = torch.randn(1, latent_dim, device=device)
    label = torch.tensor([digit], device=device)

    with torch.no_grad():
        generated_img = generator(noise, label)

    generated_img = generated_img.cpu().numpy()
    plt.imshow(generated_img[0, 0, :, :], cmap="gray")
    plt.axis('off')
    plt.show()


run = True
while run:
    user_input = input("Введіть цифру від 0 до 9 або 'exit' для виходу: ")
    if user_input.lower() == 'exit':
        run = False
        break
    elif user_input.isdigit() and int(user_input) in range(0, 10):
        generate_specific_digit(generator, int(user_input), latent_dim)
    else:
        print("Некоректний ввід. Будь ласка, введіть цифру від 0 до 9 або 'exit'")
