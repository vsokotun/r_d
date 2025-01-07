import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Получаем абсолютный путь к директории проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Добавляем корневую директорию проекта в Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wcgan.models import Generator as BaseGenerator
from wcgan.models import Discriminator as BaseDiscriminator
from models import FrequencySplitGenerator
from train import train_frequency_split_gan

def load_base_models(checkpoint_path, device):
    latent_dim = 100
    channels = 3
    
    base_generator = BaseGenerator(latent_dim, channels).to(device)
    base_discriminator = BaseDiscriminator(channels).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_generator.load_state_dict(checkpoint['generator_state_dict'])
    base_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    print("Базовые модели загружены успешно")
    return base_generator, base_discriminator

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    # Только базовая модель
    base_checkpoint_path = os.path.join(project_root, 'results/checkpoints/checkpoint_epoch_300.pt')
    print(f"Загружаем базовую модель: {base_checkpoint_path}")
    
    data_path = os.path.join(project_root, "processed_data.pt")
    batch_size = 512
    num_epochs = 100
    latent_dim = 512
    
    print(f"Using checkpoint path: {base_checkpoint_path}")
    print(f"Using data path: {data_path}")
    
    # Загружаем данные
    print("Загрузка данных...")
    data = torch.load(data_path)
    if data.dtype == torch.float16:
        data = data.float()
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Загружаем базовые модели
    base_generator, base_discriminator = load_base_models(base_checkpoint_path, device)
    
    train_frequency_split_gan(
        dataloader=dataloader,
        base_generator=base_generator,
        base_discriminator=base_discriminator,
        num_epochs=num_epochs,
        device=device,
        latent_dim=latent_dim,
        results_prefix="freq_split",
        base_checkpoint_path=base_checkpoint_path
    )

if __name__ == "__main__":
    main() 