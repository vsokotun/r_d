import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import os
import csv
from statistics import mean
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LinearLR
from torch.optim import Adam

from models import Generator, Discriminator

# Гіперпараметри
latent_dim = 100
batch_size = 1024
initial_lr = 0.0002
min_lr = 5e-5
betas = (0.0, 0.9)
n_critic = 3
lambda_gp = 10
num_epochs = 200

# Папки для збереження результатів
os.makedirs("results/images", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)

# Чекпойнт, з якого продовжуємо навчання
checkpoint_path = ''  # наразі без чекпойнту

# Пристрій та тип даних
device = torch.device("mps" if torch.mps.is_available() else "cpu")
dtype = torch.float32  # Я пробував перевести в fp16, щось перевів, щось ні, воно тут не обовʼязкове

data = torch.load("processed_data.pt")
if data.dtype == torch.float16:
    # Сам датасет також мав бути fp16
    data = data.float()
    print("Converting data to FP32 precision")
print(f"Using {data.dtype} precision")

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

generator = Generator(latent_dim=latent_dim).to(device).to(dtype)
discriminator = Discriminator().to(device).to(dtype)

g_optimizer = Adam(
    generator.parameters(), 
    lr=initial_lr, 
    betas=betas,
    weight_decay=1e-5
)

d_optimizer = Adam(
    discriminator.parameters(), 
    lr=initial_lr, 
    betas=betas,
    weight_decay=1e-5
)

# Тут я намагаюсь спершу застосувати "розігрівочний" scheduler, а далі вже реальний
warmup_epochs = 8
g_warmup = LinearLR(
    g_optimizer, 
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epochs
)

d_warmup = LinearLR(
    d_optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epochs
)

# Основний scheduler зі скидом циклу, якщо lr зайде в мінімум
T_0 = 50  # перший цикл
T_mult = 2  # Наступні цикли в 2 рази довші
g_scheduler = CosineAnnealingWarmRestarts(
    g_optimizer,
    T_0=T_0,
    T_mult=T_mult,
    eta_min=min_lr
)

d_scheduler = CosineAnnealingWarmRestarts(
    d_optimizer,
    T_0=T_0,
    T_mult=T_mult,
    eta_min=min_lr
)

start_epoch = 0
epoch_d_losses = []
epoch_g_losses = []

if checkpoint_path:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if checkpoint.get('g_optimizer_state_dict'):
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    if checkpoint.get('d_optimizer_state_dict'):
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    if checkpoint.get('g_scheduler_state_dict'):
        g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
    if checkpoint.get('d_scheduler_state_dict'):
        d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)  
    losses = checkpoint.get('losses', {'discriminator': [], 'generator': []})
    epoch_d_losses = losses['discriminator']
    epoch_g_losses = losses['generator']
    print(f"Resuming from epoch {start_epoch}")
    
    num_epochs = start_epoch + 200  # при завантаженні з чекпойнту корисно для коригування range епох

# CSV для втрат
csv_path = "results/losses.csv"
if not checkpoint_path:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'discriminator_loss', 'generator_loss'])

# градієнтний штраф
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(real_samples.size(0), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# L1 регуляризація для генератора для згладжування зображень (зайве)
def generator_regularization(gen_imgs):
    return torch.mean(torch.abs(gen_imgs[:, :, :, :-1] - gen_imgs[:, :, :, 1:])) + \
           torch.mean(torch.abs(gen_imgs[:, :, :-1, :] - gen_imgs[:, :, 1:, :]))

# тренувальний цикл
for epoch in range(start_epoch, num_epochs):
    batch_d_losses = []
    batch_g_losses = []
    
    for i, (real_imgs,) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        d_optimizer.zero_grad()
        
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs.detach())
        
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach())
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        d_optimizer.step()
        
        batch_d_losses.append(d_loss.item())
        
        if i % n_critic == 0:
            g_optimizer.zero_grad()
            
            gen_imgs = generator(z)
            fake_validity = discriminator(gen_imgs)
            
            reg_loss = 0.1 * generator_regularization(gen_imgs) 
            g_loss = -torch.mean(fake_validity) + reg_loss
            
            g_loss.backward()
            g_optimizer.step()
            
            batch_g_losses.append(g_loss.item())
        
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"[Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}]"
            )
    
    epoch_d_loss = mean(batch_d_losses)
    epoch_g_loss = mean(batch_g_losses)
    
    if epoch < warmup_epochs:
        g_warmup.step()
        d_warmup.step()
    else:
        g_scheduler.step()
        d_scheduler.step()
    
    current_lr_g = g_optimizer.param_groups[0]['lr']
    current_lr_d = d_optimizer.param_groups[0]['lr']
    
    epoch_d_losses.append(epoch_d_loss)
    epoch_g_losses.append(epoch_g_loss)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_d_loss, epoch_g_loss])
    
    with torch.no_grad():
        fake = generator(torch.randn(16, latent_dim, device=device))
        save_image(fake, f"results/images/epoch_{epoch}.png", normalize=True, nrow=4)
    
    # Кожні 5 епох зберігаємо чекпойнти
    if epoch % 5 == 0:
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': g_scheduler.state_dict(),
            'd_scheduler_state_dict': d_scheduler.state_dict(),
            'losses': {
                'discriminator': epoch_d_losses,
                'generator': epoch_g_losses
            }
        }
        torch.save(checkpoint, f"results/checkpoints/checkpoint_epoch_{epoch}.pt")
    
    print(
        f"Epoch {epoch} finished. "
        f"Average losses: D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f}, "
        f"LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f}"
    )
