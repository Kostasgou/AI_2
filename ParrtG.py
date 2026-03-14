import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from IMDBDDataset import IMDBDataset, SimpleRNN
from Logistic_Regression import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import random
import math



# Ορισμός διαδρομών δεδομένων
train_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\neg"
train_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\train\pos"
test_neg_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\neg"
test_pos_path = r"C:\Users\user\Downloads\p3210275-p3220066-p3220032\aclImdb\test\pos"


# Χρήση των ίδιων δεδομένων από το main_partA_Logistic.py
train_data = combine_data(train_neg_path, train_pos_path, 25000)
test_data = combine_data(test_neg_path, test_pos_path, 25000)

# Διαχωρισμός train σε train/dev
train_data, dev_data = train_data[:2000], train_data[2000:]

# Δημιουργία λεξιλογίου
vocab, train_labels, train_texts = make_vocab(train_data, 5000, 500, 10000)
dev_vocab, dev_labels, dev_texts = make_vocab(dev_data, 5000, 500, 10000)
test_vocab, test_labels, test_texts = make_vocab(test_data, 5000, 500, 10000)

# Δημιουργία λεξικού για το RNN
vocab_dict = {word: i+1 for i, word in enumerate(vocab)}

# Μετατροπή των δεδομένων στη μορφή που απαιτεί το IMDBDataset
train_dataset = IMDBDataset(train_texts, train_labels, vocab_dict)
dev_dataset = IMDBDataset(dev_texts, dev_labels, vocab_dict)
test_dataset = IMDBDataset(test_texts, test_labels, vocab_dict)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Αρχικοποίηση του RNN
vocab_size = len(vocab_dict) + 1
embed_dim = 100
hidden_dim = 128
num_layers = 2
dropout = 0.5
output_dim = 2
learning_rate = 0.001
epochs = 10

pretrained_embeddings = torch.randn(vocab_size, embed_dim)
model = SimpleRNN(vocab_size, embed_dim, hidden_dim, num_layers, output_dim, dropout, pretrained_embeddings)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Εκπαίδευση του RNN
train_losses = []
dev_losses = []

# Εκπαίδευση του RNN με καταγραφή του σφάλματος
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_dev_loss = 0
    
    # Εκπαίδευση
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # Υπολογισμός σφάλματος για το validation set
    model.eval()
    with torch.no_grad():
        for inputs, labels in dev_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_dev_loss += loss.item()
    
    # Μέσο σφάλμα για train και dev
    train_losses.append(total_train_loss / len(train_loader))
    dev_losses.append(total_dev_loss / len(dev_loader))
    
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Dev Loss: {dev_losses[-1]:.4f}")

# Εμφάνιση καμπυλών σφάλματος (Loss Curves)
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), dev_losses, label='Dev Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Dev Loss over Epochs')
plt.legend()
plt.show()

# Αξιολόγηση του μοντέλου για τα δεδομένα αξιολόγησης (test data)
model.eval()
y_true_rnn, y_pred_rnn = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_true_rnn.extend(labels.numpy())
        y_pred_rnn.extend(preds.numpy())

# Κατασκευή του classification report για τα δεδομένα αξιολόγησης
print("RNN Classification Report:")
print(classification_report(y_true_rnn, y_pred_rnn, zero_division=1))


# Υπολογισμός των μέσων όρων
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true_rnn, y_pred_rnn, average='micro')
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true_rnn, y_pred_rnn, average='macro')

# Εκτύπωση των μέσων όρων
print(f"\nMicro Average - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")


