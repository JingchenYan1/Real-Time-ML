import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

text = """Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology."""

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary size:", vocab_size)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

text_indices = [char_to_idx[ch] for ch in text]
text_tensor = torch.tensor(text_indices, dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, text_tensor, seq_length):
        self.text_tensor = text_tensor
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text_tensor) - self.seq_length

    def __getitem__(self, idx):
        return self.text_tensor[idx:idx + self.seq_length], self.text_tensor[idx + self.seq_length]

class CharRNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_type = rnn_type
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Unknown rnn_type")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == 'LSTM':
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_epoch(model, dataloader, criterion, optimizer):
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

def evaluate_model(model, dataloader, criterion):
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
    accuracy = correct / total
    return total_loss / len(dataloader.dataset), accuracy

results = {}
num_epochs = 20
batch_size = 32
embed_size = 64
hidden_size = 128
learning_rate = 0.001

for seq_length in [10, 20, 30]:
    dataset = TextDataset(text_tensor, seq_length)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        print(f"\nTraining model: {rnn_type}, Sequence length: {seq_length}")
        model = CharRNN(rnn_type, vocab_size, embed_size, hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            print(f"Epoch {epoch + 1:2d}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        elapsed_time = time.time() - start_time

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[(rnn_type, seq_length)] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": elapsed_time,
            "num_params": num_params
        }
        print(f"{rnn_type} Model (Sequence Length {seq_length}) Training Time: {elapsed_time:.2f} sec, Parameters: {num_params}")
        print("-" * 60)

print("\nFinal results summary:")
for key, res in results.items():
    rnn_type, seq_length = key
    print(f"Model: {rnn_type:3s}, Sequence Length: {seq_length:2d}, Train Loss: {res['train_loss']:.4f}, "
          f"Val Loss: {res['val_loss']:.4f}, Val Acc: {res['val_acc']:.4f}, "
          f"Time: {res['time']:.2f}s, Params: {res['num_params']}")
