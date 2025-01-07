import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out + identity)
        
        return out

class CarFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Сеть для выделения ключевых частей автомобиля
        self.segmentation = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 4, 1)  # 4 маски: капот, кузов, колеса, детали
        )
        
        # Энкодер для извлечения фич из каждой части
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(64, 64)
        )

    def forward(self, x):
        masks = torch.sigmoid(self.segmentation(x))  # B x 4 x H x W
        features = self.feature_encoder(x)  # B x 64 x H x W
        return masks, features

class FrequencySplitGenerator(nn.Module):
    def __init__(self, base_generator):
        super().__init__()
        self.base_generator = base_generator
        
    def forward(self, noise, real_samples):
        # Генерируем изображение
        fake_imgs = self.base_generator(noise)
        
        # Извлекаем фичи
        real_features = self.base_generator.get_features(real_samples)
        fake_features = self.base_generator.get_features(fake_imgs)
        
        # Ищем похожие части
        similar_parts = self.find_similar_parts(real_features, fake_features)
        
        # Смешиваем фичи
        result = self.mix_features(fake_imgs, similar_parts)
        
        return result

    def find_similar_parts(self, base_features, real_features, real_masks):
        # Преобразуем фичи в плоский вектор для каждого изображения
        base_features_flat = base_features.view(base_features.size(0), -1)  # [B, C*H*W]
        real_features_flat = real_features.view(real_features.size(0), -1)  # [B, C*H*W]
        
        # Нормализуем векторы
        base_features_norm = F.normalize(base_features_flat, p=2, dim=1)
        real_features_norm = F.normalize(real_features_flat, p=2, dim=1)
        
        # Считаем косинусное сходство
        similarity = torch.mm(base_features_norm, real_features_norm.t())  # [B, B]
        
        # Находим топ-3 похожих примера для каждого изображения в батче
        _, indices = similarity.topk(k=3, dim=1)  # [B, 3]
        
        return indices

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.main = nn.Sequential(
            # Начальный слой
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Средние слои
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Финальные слои
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Flatten(0, -1)  # Убираем
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)  # Явно задаем размерность [batch_size, 1] 