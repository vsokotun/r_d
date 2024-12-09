import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

batch_size=128
lr = 0.0003
num_epochs = 100

writer = SummaryWriter()

def load_dataset():
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset


def analyze_dataset(dataset):
    labels = []

    for _, label in dataset:
        labels.append(label)

    labels = torch.tensor(labels)

    missing_data = any(item is None for item in dataset)

    if missing_data:
        print("Набір даних має пропущені елементи\n")
    else:
        print("Всі дані повні.\n")

    unique, counts = torch.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Клас {label.item()+1}: {count.item()} елементів")

a_trainset, a_testset = load_dataset()
analyze_dataset(dataset=iter(a_trainset))

def calculate_mean_and_std(loader):
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        total_pixels += batch_pixels
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_squared_sum / total_pixels - mean ** 2)
    return mean, std

a_train_loader = DataLoader(a_trainset, batch_size=10000, shuffle=False)
mean, std = calculate_mean_and_std(a_train_loader)

print(mean, std)

def load_full_dataset():
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset


n_train, n_test = load_full_dataset()
n_train_loader = DataLoader(n_train, batch_size=batch_size, shuffle=True)
n_test_loader = DataLoader(n_test, batch_size=batch_size, shuffle=False)

for images, _ in n_train_loader:
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    print(f"Перевірка нормалізації: середнє {mean}, стандартне відхилення {std}")
    break


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc(x)
        return x


device = 'mps'
model = CNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for i, (data, target) in enumerate(n_train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(n_train_loader)
    train_losses.append(avg_train_loss)

    writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(n_train_loader) + i)

    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in n_test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_func(output, target)
            total_val_loss += loss.item()

            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    avg_val_loss = total_val_loss / len(n_test_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100.0 * correct / total
    val_accuracies.append(val_accuracy)

    writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

writer.close()