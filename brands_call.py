import torch
import torch.nn as nn

def tokenize_sequence(sequence, char_to_idx, max_length=30):
    tokens = [char_to_idx[c] for c in sequence if c in char_to_idx]
    tokens = tokens[:max_length] + [0] * (max_length - len(tokens))
    return tokens

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

def load_brand_model(model_path='brand_classification_model.pth'):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    brand_to_idx = checkpoint['brand_to_idx']
    idx_to_brand = {v: k for k, v in brand_to_idx.items()}

    vocab_size = len(char_to_idx)
    output_dim = len(brand_to_idx)
    model = BrandPredictionModel(vocab_size, embedding_dim=64, hidden_dim=128, output_dim=output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, char_to_idx, idx_to_char, idx_to_brand

def predict_brand(model, initial_context, char_to_idx, idx_to_brand, max_length=30):
    context_tokens = tokenize_sequence(initial_context, char_to_idx, max_length)
    context_tensor = torch.tensor([context_tokens])

    with torch.no_grad():
        logits = model(context_tensor)
        predicted_idx = logits.argmax(dim=1).item()
        predicted_brand = idx_to_brand[predicted_idx]

    return predicted_brand


"""
Тестовий запуск.
По суті щось подібне буде запущено на сервері Flask в контейнері,
для передачі автозаповнення в додаток.
"""

if __name__ == "__main__":
    model, char_to_idx, idx_to_char, idx_to_brand = load_brand_model('brand_classification_model.pth')

    # Цикл для пользовательского ввода
    while True:
        user_input = input("Type a symbol to get a prediction or type 'exit' to finish: ").strip()
        if user_input.lower() == 'exit':
            break
        predicted = predict_brand(model, user_input.lower(), char_to_idx, idx_to_brand)
        if predicted:
            if user_input.lower() in predicted.lower(): # Так як марки з надто низькою частотою модель видає погано, перевіряємо чи відповідає інпут предикціям
                print(predicted)
            else:
                continue
