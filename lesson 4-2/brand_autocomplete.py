import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Підготовка даних
def preprocess_brands_with_frequency(df):
    """
    Для коректної роботи застосунка приведемо ввідні дані до lowercase.
    В подальшому, приймаючи запит від користувача, будемо переводити його
    до нижнього регістру. Втім модель має відповідати "правильним" чином,
    з урахуванням регістру.
    """
    examples = []
    # Створюємо індекси класів
    brand_to_idx = {brand: idx for idx, brand in enumerate(df['Brand'].unique())}

    for _, row in df.iterrows():
        brand = row['Brand'] # Залишаємо стандарте написання для результатів
        lower_brand = brand.lower()  # Контекст переводимо в нижній регістр

        # Створюємо префікси для марок. Таким чином, щоб v, vo, vol, volv і т.д. могло бути класифіковано як Volvo
        for i in range(1, len(lower_brand) + 1):
            context = lower_brand[:i]
            target = brand_to_idx[brand] # Привʼязуємо префікс одразу до індексу класу.
            examples.append((context, target))

    return examples, brand_to_idx


# Клас для обробки вхідних даних та створення датасету

class BrandDataset(Dataset):
    def __init__(self, examples, char_to_idx):
        self.examples = examples
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target = self.examples[idx]
        context_idx = [self.char_to_idx[char] for char in context]
        return torch.tensor(context_idx, dtype=torch.long), target

# Модель
class BrandPredictionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=30):
        super(BrandPredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

"""
Завантажуємо дані
Оригінальне похождення - з бази даних компанії
По суті просто перелік марок та моделей авто, які зʼявлялись в базі.
Список не зведено до унікальних значень для збереження оригінального співвідношення даних,
на основі якого має бути виведена частотність, і по суті прийняте рішення, чи пропонувати цей варіант
"""

df = pd.read_csv('car_data.csv')
all_text = ' '.join(df['Brand'].str.lower())
chars = sorted(set(all_text))
chars = ['<PAD>'] + chars
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

examples, brand_to_idx = preprocess_brands_with_frequency(df)
dataset = BrandDataset(examples, char_to_idx)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, len(char_to_idx)))

def collate_fn(batch, pad_idx):
    contexts, targets = zip(*batch)
    seq_lens = [len(context) for context in contexts]
    max_len = max(seq_lens)
    padded_contexts = torch.zeros(len(contexts), max_len, dtype=torch.long).fill_(pad_idx)

    for i, context in enumerate(contexts):
        padded_contexts[i, :seq_lens[i]] = context

    return padded_contexts, torch.tensor(targets, dtype=torch.long)

vocab_size = len(chars)
output_dim = len(brand_to_idx)
device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
model = BrandPredictionModel(vocab_size, embedding_dim=64, hidden_dim=128, output_dim=output_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Функція навчання. По суті достатньо двух епох, три для вірності. Набір даних досить обмежений, машина не може навчитись більшому

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for context, target in dataloader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted = output.argmax(dim=1)
        correct = (predicted == target).to(dtype=torch.float).sum().item()
        total_accuracy += correct / len(target)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'brand_to_idx': brand_to_idx,
}, 'brand_classification_model.pth')
