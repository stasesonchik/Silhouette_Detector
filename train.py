import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

from generation_process_new.model_config import DEVICE, BATCH_SIZE, EPOCHS, EMBEDDING_DIR, ATT_CLASSES, FOCAL_WEIGHTS
from generation_process_new.model import VisionAttrTransformer


# ===== Dataset & Collate =====
class EmbeddingAttrDataset(Dataset):
    def __init__(self, embedding_dir):
        self.files = [
            os.path.join(embedding_dir, f)
            for f in os.listdir(embedding_dir) if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        emb = data["embedding"].to(DEVICE).float()
        labels = data["labels"]
        return emb, labels


def collate_fn(batch):
    embs, labels = zip(*batch)
    max_len = max(e.size(0) for e in embs)
    padded, masks = [], []
    for e in embs:
        pad = max_len - e.size(0)
        if pad > 0:
            e = torch.cat([e, torch.zeros(pad, *e.shape[1:], device=DEVICE)], dim=0)
        padded.append(e)
        masks.append(torch.cat([
            torch.ones(e.size(0), device=DEVICE, dtype=torch.bool),
            torch.zeros(pad, device=DEVICE, dtype=torch.bool)
        ]))
    return torch.stack(padded), labels, torch.stack(masks)


# ===== Focal Loss =====
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ===== Train & Validate =====
def train_one_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0


    all_preds = {a: [] for a in ATT_CLASSES}
    all_trues = {a: [] for a in ATT_CLASSES}
    criteria = {a: FocalLoss(alpha=FOCAL_WEIGHTS[a], gamma=2) for a in ATT_CLASSES}

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for embs, labels, mask in pbar:
        out = model(embs, mask)


        loss = 0.0
        for attr, logits in out.items():
            y_true = torch.tensor([l[attr] for l in labels], device=DEVICE)
            loss += criteria[attr](logits, y_true)
            y_pred = torch.argmax(logits, dim=1)
            all_trues[attr].extend(y_true.cpu().tolist())
            all_preds[attr].extend(y_pred.cpu().tolist())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    avg_loss = total_loss / len(loader)
    print(f"\nTrain Loss: {avg_loss:.4f}")


    print("=== Train Classification Report ===")
    for attr in ATT_CLASSES:
        print(f"--- {attr} ---")
        print(classification_report(
            all_trues[attr],
            all_preds[attr],
            labels=list(range(ATT_CLASSES[attr])),
            digits=3
        ))

    macro_f1 = 0.0
    for attr in ATT_CLASSES:
        report = classification_report(
            all_trues[attr],
            all_preds[attr],
            labels=list(range(ATT_CLASSES[attr])),
            output_dict=True
        )
        macro_f1 += report['macro avg']['f1-score']
    macro_f1 /= len(ATT_CLASSES)

    return avg_loss, macro_f1


def validate(model, loader):
    model.eval()
    total_loss = 0.0
    preds = {a: [] for a in ATT_CLASSES}
    trues = {a: [] for a in ATT_CLASSES}
    criteria = {a: FocalLoss(alpha=FOCAL_WEIGHTS[a], gamma=2) for a in ATT_CLASSES}

    with torch.no_grad():
        for embs, labels, mask in tqdm(loader, desc="Validating"):
            out = model(embs, mask)

            for attr, logits in out.items():
                y_true = torch.tensor([l[attr] for l in labels], device=DEVICE)
                total_loss += criteria[attr](logits, y_true).item()
                y_pred = torch.argmax(logits, dim=1)
                trues[attr].extend(y_true.cpu().tolist())
                preds[attr].extend(y_pred.cpu().tolist())

    avg_loss = total_loss / len(loader)
    print(f"\nVal Loss: {avg_loss:.4f}")

    print("=== Validation Classification Report ===")
    macro_f1 = 0.0
    for attr in ATT_CLASSES:
        print(f"--- {attr} ---")
        report_dict = classification_report(
            trues[attr],
            preds[attr],
            labels=list(range(ATT_CLASSES[attr])),
            digits=3,
            output_dict=True
        )

        print(classification_report(
            trues[attr],
            preds[attr],
            labels=list(range(ATT_CLASSES[attr])),
            digits=3
        ))
        macro_f1 += report_dict['macro avg']['f1-score']
    macro_f1 /= len(ATT_CLASSES)

    return avg_loss, macro_f1


if __name__ == "__main__":
    ds = EmbeddingAttrDataset(EMBEDDING_DIR)
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(
        ds,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = VisionAttrTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    total_steps = EPOCHS * len(train_loader)
    warmup = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: float(step) / warmup if step < warmup
                     else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    )

    best_f1 = 0.0
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    for epoch in range(EPOCHS):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)

        val_loss, val_f1 = validate(model, test_loader)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "hidden_dim": model.input_proj.out_features,
                    "num_heads": model.layers[0].self_attn.num_heads,
                    "num_layers": len(model.layers),
                    "attr_sizes": ATT_CLASSES
                }
            }, "best_checkpoint.pt")
            print(f"*** Saved best model at epoch {epoch+1} with val_macro_f1={best_f1:.4f} ***\n")

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("training_history.png")

    plt.figure()
    plt.plot(history["train_f1"], label="train macro F1")
    plt.plot(history["val_f1"], label="val macro F1")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.savefig("f1_history.png")
