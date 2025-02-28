import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import Dataset, DataLoader
import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = [char_to_int[ch] for ch in text]


def create_sequences_targets(encoded_text, sequence_length):
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i + sequence_length]
        target = encoded_text[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return sequences, targets


class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]


class CharModel(nn.Module):
    def __init__(self, model_type, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(CharModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.model_type = model_type
        if model_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be LSTM or GRU")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        if self.model_type == "LSTM":
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def generate_text(model, seed_text, gen_length, device):
    model.eval()
    generated = seed_text
    input_seq = torch.tensor([char_to_int[ch] for ch in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(gen_length):
            output = model(input_seq)
            probs = torch.softmax(output, dim=1).squeeze()
            idx = torch.multinomial(probs, 1).item()
            generated += int_to_char[idx]
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[idx]], dtype=torch.long).to(device)], dim=1)
    return generated


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 128
num_epochs = 10
learning_rate = 0.001
embed_dim = 128
hidden_size = 256
num_layers = 2

results = {}

for seq_length in [20, 30, 50]:
    print(f"\n=== Sequence Length = {seq_length} ===")
    sequences, targets = create_sequences_targets(encoded_text, seq_length)
    dataset = CharDataset(sequences, targets)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    for model_type in ["LSTM", "GRU"]:
        print(f"\n--- Training Model: {model_type} ---")
        model = CharModel(model_type, vocab_size, embed_dim, hidden_size, num_layers=num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
            perplexity = math.exp(val_loss)
            print(f"Epoch {epoch + 1:2d}/{num_epochs}, Train Loss: {train_loss:.4f}, " +
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Perplexity: {perplexity:.2f}")
        elapsed_time = time.time() - start_time

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[(model_type, seq_length)] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "perplexity": perplexity,
            "time": elapsed_time,
            "num_params": num_params
        }
        print(
            f"{model_type} Model (Sequence Length {seq_length}) Training Time: {elapsed_time:.2f} sec, Parameters: {num_params}")

        seed = "The "
        generated = generate_text(model, seed, 100, device)
        print("Generated Text Example:")
        print(generated)
        print("-" * 60)

print("\n=== Experiment Results Summary ===")
for key, res in results.items():
    model_type, seq_length = key
    print(f"Model: {model_type:4s}, Sequence Length: {seq_length:2d}, Final Train Loss: {res['train_loss']:.4f}, " +
          f"Final Val Loss: {res['val_loss']:.4f}, Val Acc: {res['val_acc']:.4f}, " +
          f"Perplexity: {res['perplexity']:.2f}, Time: {res['time']:.2f}s, Params: {res['num_params']}")
