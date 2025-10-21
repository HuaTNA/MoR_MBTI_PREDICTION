

import os, torch, copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.optim as optim

# ============================= Dataset =============================
class MBTIDataset(Dataset):
    def __init__(self, root, tokenizer, classes, max_len=128):
        self.samples, self.targets, self.classes = [], [], classes
        for idx, c in enumerate(classes):
            folder = os.path.join(root, c)
            for f in os.listdir(folder):
                if f.endswith(".txt"):
                    with open(os.path.join(folder, f), "r", encoding="utf-8") as fp:
                        text = fp.read().strip()
                        self.samples.append(text)
                        self.targets.append(idx)
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        text, label = self.samples[i], self.targets[i]
        enc = self.tok(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, label

# ============================= MoR Router =============================
class MoRRouter1D(nn.Module):
    """Simple recursive gating layer for text features."""
    def __init__(self, hidden, depth=2):
        super().__init__()
        self.depth = depth
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        for _ in range(self.depth):
            g = self.gate(x)
            x = x + g * x
        return x

# ============================= Model =============================
class TextMoRModel(nn.Module):
    """BERT + MoR recursive gating + classifier."""
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden = self.bert.config.hidden_size
        self.mor = MoRRouter1D(hidden, depth=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        out = self.mor(out)
        return self.fc(out)

# ============================= Train & Validate =============================
def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0
    for batch in tqdm(loader, desc="Train" if train else "Val"):
        input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch[1].to(device) if isinstance(batch, tuple) else batch['labels'].to(device) if 'labels' in batch else batch['label'].to(device)
        with autocast(device_type='cuda'):
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        loss_sum += loss.item() * labels.size(0)
    return loss_sum / total, correct / total

def main():
    data_root = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\TextData"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    classes = sorted(os.listdir(os.path.join(data_root, 'train')))

    train_loader = DataLoader(MBTIDataset(os.path.join(data_root, "train"), tokenizer, classes), batch_size=8, shuffle=True)
    val_loader   = DataLoader(MBTIDataset(os.path.join(data_root, "val"), tokenizer, classes), batch_size=8)
    test_loader  = DataLoader(MBTIDataset(os.path.join(data_root, "test"), tokenizer, classes), batch_size=8)

    model = TextMoRModel(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    best_acc = 0
    for epoch in range(4):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        print(f"[{epoch+1}] train_acc={ta:.4f} val_acc={va:.4f}")
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), os.path.join(data_root, "text_mor_best.pth"))

    print("Best val acc:", best_acc)
    _, test_acc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    main()
