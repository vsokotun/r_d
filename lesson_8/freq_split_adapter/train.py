import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from datetime import datetime
from models import FrequencySplitGenerator, Discriminator
from tqdm import tqdm  # Добавляем прогресс-бар
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_image_grid(images, path, nrow=5, normalize=True):
    """Сохраняет сетку изображений"""
    vutils.save_image(images, path, nrow=nrow, normalize=normalize)

def train_frequency_split_gan(
    dataloader,
    base_generator,
    base_discriminator,
    num_epochs=20,
    device="mps",
    latent_dim=512,
    results_prefix="freq_split",
    start_epoch=0,
    base_checkpoint_path=None,
    adapter_checkpoint_path=None
):
    # Создаем директории для результатов
    samples_dir = f'results/{results_prefix}_samples'
    checkpoints_dir = f'results/{results_prefix}_checkpoints'
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Создаем модели
    generator = FrequencySplitGenerator(base_generator).to(device)
    discriminator = base_discriminator
    
    # Оптимизаторы
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Начинаем с нуля
    losses = {
        'g_losses': [],
        'd_losses': [],
        'epochs': [],
        'batches': []
    }
    print("Начинаем обучение с нуля")
    
    # Создаем фиксированные тензоры для тестирования
    fixed_noise = torch.randn(25, latent_dim, device=device)
    fixed_real_imgs = next(iter(dataloader))[0][:25].to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Отключаем все принты
    generator.verbose = False
    torch.set_printoptions(profile="default")
    
    print(f"Продолжаем с эпохи {start_epoch}")
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(dataloader, desc=f'Эпоха {epoch}')
        for i, batch in enumerate(pbar):
            # Преобразуем список в тензор
            real_imgs = torch.stack(batch).to(device)
            batch_size = real_imgs.size(0)
            
            # Генерируем случайный шум
            noise = torch.randn(batch_size, latent_dim).to(device)
            
            # Генерируем фейковые изображения
            fake_imgs = generator(noise, real_imgs)
            
            # Создаем метки для текущего батча
            real_label_tensor = torch.ones(batch_size, device=device) * 0.9  # Label smoothing
            fake_label_tensor = torch.zeros(batch_size, device=device) + 0.1  # Label smoothing
            
            # Обучаем дискриминатор
            if i % 2 == 0:
                optimizer_D.zero_grad(set_to_none=True)
                d_real = discriminator(real_imgs)
                d_fake = discriminator(fake_imgs.detach())
                
                d_loss_real = criterion(d_real.view(-1), real_label_tensor)
                d_loss_fake = criterion(d_fake.view(-1), fake_label_tensor)
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
                d_loss = d_loss_real + d_loss_fake + 5 * gradient_penalty
                d_loss.backward(retain_graph=True)
                optimizer_D.step()
            
            # Обучаем генератор
            optimizer_G.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake_imgs).view(-1), real_label_tensor)
            g_loss.backward()
            optimizer_G.step()
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss.item():.3f}'
            })
            
            # Сохраняем значения потерь
            losses['g_losses'].append(g_loss.item())
            losses['d_losses'].append(d_loss.item())
            losses['epochs'].append(epoch)
            losses['batches'].append(i)
            
            # Сохраняем изображения каждые 100 батчей
            if i % 100 == 0:
                with torch.no_grad():
                    fake_fixed = generator(fixed_noise, fixed_real_imgs)
                    save_image_grid(fake_fixed, 
                                  f'{samples_dir}/fake_epoch_{epoch}_batch_{i}.png')
            
            # Сохраняем тестовые изображения каждые 100 батчей
            if i % 100 == 0:
                with torch.no_grad():
                    fake_fixed = generator(fixed_noise, fixed_real_imgs)
                    save_image_grid(fake_fixed, 
                                  f'{samples_dir}/fake_epoch_{epoch}_batch_{i}.png')
                    
                    current_fake = generator(torch.randn(25, latent_dim, device=device), real_imgs[:25])
                    save_image_grid(current_fake, 
                                  f'{samples_dir}/current_fake_epoch_{epoch}_batch_{i}.png')
        
        # Выводим средние потери за эпоху
        avg_g_loss = sum(losses['g_losses']) / len(losses['g_losses'])
        avg_d_loss = sum(losses['d_losses']) / len(losses['d_losses'])
        print(f'[Epoch {epoch}/{num_epochs}] G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')
        
        # Сохраняем чекпоинт с историей потерь
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': optimizer_G.state_dict(),
                'd_optimizer_state_dict': optimizer_D.state_dict(),
                'losses': losses,  # Добавляем историю потерь
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }
            torch.save(
                checkpoint,
                f'{checkpoints_dir}/checkpoint_epoch_{epoch+1}.pt'
            )
            
            # Сохраняем график потерь
            plt.figure(figsize=(10, 5))
            plt.plot(losses['g_losses'], label='Generator Loss')
            plt.plot(losses['d_losses'], label='Discriminator Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{samples_dir}/losses_epoch_{epoch+1}.png')
            plt.close()
    
    # Сохраняем финальный чекпоинт
    checkpoint = {
        'epoch': num_epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': optimizer_G.state_dict(),
        'd_optimizer_state_dict': optimizer_D.state_dict(),
        'losses': losses,  # Добавляем историю потерь
        'g_loss': avg_g_loss,
        'd_loss': avg_d_loss
    }
    torch.save(
        checkpoint,
        f'{checkpoints_dir}/checkpoint_final.pt'
    )
    
    # Сохраняем финальный график потерь
    plt.figure(figsize=(10, 5))
    plt.plot(losses['g_losses'], label='Generator Loss')
    plt.plot(losses['d_losses'], label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{samples_dir}/losses_final.png')
    plt.close()
    
    # Сохраняем финальные результаты
    with torch.no_grad():
        final_fake = generator(fixed_noise, fixed_real_imgs)
        save_image_grid(final_fake, f'{samples_dir}/final_fake.png')
    
    return generator, discriminator

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    # Изменяем размерность fake для соответствия с выходом дискриминатора
    fake = torch.ones(d_interpolates.size(), device=real_samples.device)  # Теперь размерность совпадает
    
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