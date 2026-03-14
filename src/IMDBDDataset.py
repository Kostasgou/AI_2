import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=500):
        self.data = [self.tokenize(text, vocab, max_len) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def tokenize(self, text, vocab, max_len=500):
        tokenized = [vocab.get(word, 0) for word in text.split()[:max_len]]
        padded = tokenized + [0] * (max_len - len(tokenized))
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define Simple RNN Model
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, dropout, pretrained_embeddings):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        pooled = torch.max(rnn_out, dim=1)[0]  # Global Max Pooling
        out = self.fc(self.dropout(pooled))
        return out