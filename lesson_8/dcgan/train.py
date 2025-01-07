import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import os
import csv
from statistics import mean
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from models import Generator, Discriminator
from config import *

os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
dtype = torch.float32

data = torch.load("processed_data.pt")
if data.dtype == torch.float16:
    data = data.float()
print(f"Using {data.dtype} precision")

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

generator = Generator(latent_dim=LATENT_DIM, feature_maps=FEATURE_MAPS).to(device)
discriminator = Discriminator(feature_maps=FEATURE_MAPS).to(device)

g_optimizer = AdamW(generator.parameters(), lr=INITIAL_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
d_optimizer = AdamW(discriminator.parameters(), lr=INITIAL_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

g_scheduler = CosineAnnealingLR(g_optimizer, T_max=SCHEDULER_T_MAX, eta_min=MIN_LR)
d_scheduler = CosineAnnealingLR(d_optimizer, T_max=SCHEDULER_T_MAX, eta_min=MIN_LR)

criterion = nn.BCELoss()

def get_labels(size, real=True):
    if real:
        return torch.full((size,), REAL_LABEL_VAL, device=device)
    return torch.full((size,), FAKE_LABEL_VAL, device=device)

with open(LOSSES_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'discriminator_loss', 'generator_loss'])

for epoch in range(NUM_EPOCHS):
    batch_d_losses = []
    batch_g_losses = []
    
    for i, (real_imgs,) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        d_optimizer.zero_grad()
        
        label_real = get_labels(batch_size, real=True)
        label_fake = get_labels(batch_size, real=False)
        
        output_real = discriminator(real_imgs)
        d_loss_real = criterion(output_real, label_real)
        
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = generator(noise)
        output_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        g_optimizer.zero_grad()
        
        output_fake = discriminator(fake_imgs)
        g_loss = criterion(output_fake, label_real)
        
        g_loss.backward()
        g_optimizer.step()
        
        batch_d_losses.append(d_loss.item())
        batch_g_losses.append(g_loss.item())
        
        if i % LOG_FREQ == 0:
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] "
                f"[Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}]"
            )
    
    g_scheduler.step()
    d_scheduler.step()
    
    epoch_d_loss = mean(batch_d_losses)
    epoch_g_loss = mean(batch_g_losses)
    
    with open(LOSSES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_d_loss, epoch_g_loss])
    
    if epoch % SAMPLE_FREQ == 0:
        with torch.no_grad():
            test_noise = torch.randn(16, LATENT_DIM, device=device)
            fake = generator(test_noise)
            save_image(fake, f"{IMAGES_PATH}/epoch_{epoch}.png", normalize=True, nrow=4)
    
    if epoch % CHECKPOINT_FREQ == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': g_scheduler.state_dict(),
            'd_scheduler_state_dict': d_scheduler.state_dict(),
        }, f"{CHECKPOINTS_PATH}/checkpoint_epoch_{epoch}.pt")
    
    print(
        f"Epoch {epoch} finished. "
        f"Average losses: D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f}"
    ) 