import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
import random

# === ПАРАМЕТРЫ ===
device = "cuda:0"
embedding_dir = "embeddings"
batch_size = 64
epochs = 10

# === КЛАСС ДАТАСЕТА ===
class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.files = [os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) if f.endswith(".pt")]
        self.valid_files = []
        for f in self.files:
            try:
                data = torch.load(f)
                if data["embedding"].shape == (3584,):
                    self.valid_files.append(f)
            except Exception:
                continue

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        item = torch.load(self.valid_files[idx])
        embedding = item["embedding"].to(torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)
        return embedding, label

# === КЛАССИФИКАТОР ===
class GenderClassifier(nn.Module):
    def __init__(self, input_dim=3584, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.classifier(x)

# === ДАТАСЕТ И РАЗДЕЛЕНИЕ ===
full_dataset = EmbeddingDataset(embedding_dir)
train_len = int(0.8 * len(full_dataset))
val_len = int(0.1 * len(full_dataset))
test_len = len(full_dataset) - train_len - val_len

train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# === ОБУЧЕНИЕ ===
clf = GenderClassifier().to(device)
optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    clf.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = clf(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_losses.append(total_loss)
    train_accuracies.append(train_acc)

    # === ВАЛИДАЦИЯ ===
    clf.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

# === ОЦЕНКА НА TEST ===
clf.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = clf(x).argmax(dim=1)
        test_correct += (preds == y).sum().item()
        test_total += y.size(0)

test_acc = test_correct / test_total
print(f"\n=== Final Test Accuracy: {test_acc:.4f} ===")

# === ГРАФИКИ ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
