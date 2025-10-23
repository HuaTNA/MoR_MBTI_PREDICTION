"""
4.3MoR_Text_Optimized.py
Optimized Text MBTI classifier with BERT + MoR Router (adaptive refinement placeholder)
- Gradual Unfreezing (stage-wise)
- OneCycleLR scheduling
- AMP + TF32 acceleration
- Label Smoothing CrossEntropy
- Best checkpoint + confusion matrix + training log

Folder layout (per split):
DATA_ROOT/
  train/<ClassName>/*.txt
  val/<ClassName>/*.txt
  test/<ClassName>/*.txt
"""

import os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
hf_logging.set_verbosity_error()  # keep console clean

# ==================== Global Performance Tweaks ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = "cuda"
AMP_ENABLED = DEVICE.type == "cuda"

# ============================= Config =============================
DATA_ROOT = r"C:\APS360_project\MoR_MBTI_PREDICTION\data\TextData"  # set your data root here

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 160
BATCH_SIZE = 8
EPOCHS = 12
WARM_FREEZE_EPOCHS = 2       # stage-1: freeze all
PARTIAL_UNFREEZE_EPOCHS = 4  # stage-2: unfreeze last 4 layers (total first 6 epochs)
INIT_LR = 2e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTH = 0.1
GRAD_CLIP = 1.0
SEED = 42

# ============================= Utils =============================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_classes(split_dir):
    return sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

def plot_confusion(preds, labels, classes, save_path):
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================= Dataset =============================
class MBTIDataset(Dataset):
    """
    Reads .txt files from class folders. Each file is one sample.
    Tokenization is done on-the-fly for flexibility (simplifies DataLoader).
    """
    def __init__(self, root, classes, tokenizer, max_len=MAX_LEN, show_progress=True):
        self.samples, self.targets = [], []
        self.classes = classes
        self.tok = tokenizer
        self.max_len = max_len
        split_name = os.path.basename(os.path.normpath(root))
        file_index = []
        for idx, c in enumerate(classes):
            folder = os.path.join(root, c)
            if not os.path.isdir(folder): continue
            for f in os.listdir(folder):
                if f.lower().endswith(".txt"):
                    file_index.append((os.path.join(folder, f), idx))

        iterator = file_index
        if show_progress and file_index:
            iterator = tqdm(file_index, desc=f"Loading {split_name}", unit="file")

        for path, label_idx in iterator:
            with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read().strip()
            if text:
                self.samples.append(text)
                self.targets.append(label_idx)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        text, label = self.samples[i], self.targets[i]
        enc = self.tok(text,
                       padding='max_length',
                       truncation=True,
                       max_length=self.max_len,
                       return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# ============================= MoR Router =============================
class MoRRouter1D(nn.Module):
    """
    Text MoR router (placeholder):
    - Gate on [CLS] decides whether to apply an extra tiny encoder step.
    - We use a single TransformerEncoderLayer as a lightweight refinement.
    - Early exit if gate confidence is low (adaptive recursion).
    """
    def __init__(self, hidden_dim, nhead=12, depth=2, threshold=0.55):
        super().__init__()
        self.depth = depth
        self.threshold = threshold
        self.cls_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()
        )
        self.extra = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                batch_first=True, dim_feedforward=4*hidden_dim,
                                                activation="gelu", dropout=0.1)

    def forward(self, h):  # h: [B,T,H]
        for _ in range(self.depth):
            cls = h[:, 0, :]                 # [CLS]
            prob = self.cls_gate(cls).view(-1, 1, 1)  # [B,1,1]
            if prob.mean() < self.threshold: # adaptive early exit
                break
            refined = self.extra(h)
            h = prob * refined + (1 - prob) * h
        return h

# ============================= Model =============================
class TextMoRModel(nn.Module):
    """
    BERT encoder + MoR recursion over token sequence + classification on [CLS].
    Gradual unfreezing is handled outside via .requires_grad flags.
    """
    def __init__(self, num_classes, model_name=MODEL_NAME, use_mor=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.use_mor = use_mor
        self.mor = MoRRouter1D(h) if use_mor else None
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(h, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,T,H]
        if self.use_mor:
            h = self.mor(h)
        cls = h[:, 0, :]
        return self.fc(self.dropout(cls))

# ============================= Train / Eval =============================
def train_epoch(model, loader, criterion, optimizer, scaler, scheduler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=AMP_ENABLED):
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(logits, batch['labels'])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        preds = logits.argmax(1)
        total += batch['labels'].size(0)
        correct += preds.eq(batch['labels']).sum().item()
        loss_sum += loss.item() * batch['labels'].size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    preds_all, labels_all = [], []
    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=AMP_ENABLED):
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(logits, batch['labels'])
        preds = logits.argmax(1)
        total += batch['labels'].size(0)
        correct += preds.eq(batch['labels']).sum().item()
        loss_sum += loss.item() * batch['labels'].size(0)
        preds_all += preds.cpu().tolist()
        labels_all += batch['labels'].cpu().tolist()
    return loss_sum / total, correct / total, preds_all, labels_all

# ============================= Unfreezing Helpers =============================
def freeze_all_bert(model: TextMoRModel):
    for p in model.encoder.parameters():
        p.requires_grad = False

def unfreeze_last_n_layers(model: TextMoRModel, n=4):
    enc = model.encoder.encoder
    layers = enc.layer
    for p in model.encoder.embeddings.parameters():
        p.requires_grad = False
    for i in range(len(layers)-n, len(layers)):
        for p in layers[i].parameters():
            p.requires_grad = True

def unfreeze_all(model: TextMoRModel):
    for p in model.parameters():
        p.requires_grad = True

def param_groups(model: TextMoRModel, base_lr=INIT_LR):
    # Higher LR for classifier & MoR; lower for encoder
    groups = []
    head = list(model.fc.parameters()) + list(model.dropout.parameters())
    if model.use_mor:
        head += list(model.mor.parameters())
    groups.append({"params": head, "lr": base_lr})
    groups.append({"params": model.encoder.parameters(), "lr": base_lr * 0.25})
    return groups

# ============================= Main =============================
def main():
    set_seed()
    os.makedirs(DATA_ROOT, exist_ok=True)
    classes = list_classes(os.path.join(DATA_ROOT, "train"))
    print(f"Classes ({len(classes)}): {classes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    mk_loader = lambda split, tf_batch: DataLoader(
        MBTIDataset(os.path.join(DATA_ROOT, split), classes, tokenizer),
        batch_size=tf_batch,
        shuffle=(split == "train"),
        num_workers=4 if AMP_ENABLED else 0,
        pin_memory=AMP_ENABLED,
    )
    train_loader = mk_loader("train", BATCH_SIZE)
    val_loader   = mk_loader("val",   BATCH_SIZE)
    test_loader  = mk_loader("test",  BATCH_SIZE)

    model = TextMoRModel(num_classes=len(classes), model_name=MODEL_NAME, use_mor=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Stage 1: freeze all encoder (warmup)
    freeze_all_bert(model)
    optimizer = optim.AdamW(param_groups(model, INIT_LR), weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=INIT_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    scaler = GradScaler(device=AMP_DEVICE_TYPE, enabled=AMP_ENABLED)

    best_acc, best_state = 0.0, None
    log_file = os.path.join(DATA_ROOT, "text_train_log.txt")

    for epoch in range(EPOCHS):
        # Stage switching
        if epoch == WARM_FREEZE_EPOCHS:
            print(">>> Unfreeze last 4 layers of BERT")
            unfreeze_last_n_layers(model, n=4)
        if epoch == PARTIAL_UNFREEZE_EPOCHS:
            print(">>> Unfreeze ALL BERT layers")
            unfreeze_all(model)

        tl, ta = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
        vl, va, _, _ = eval_epoch(model, val_loader, criterion)
        print(f"[{epoch+1:02d}] train_acc={ta:.4f} | val_acc={va:.4f}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{ta:.4f}\t{va:.4f}\n")

        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
            save_path = os.path.join(DATA_ROOT, "text_mor_best.pth")
            torch.save({'model': best_state, 'classes': classes, 'epoch': epoch, 'val_acc': va}, save_path)
            print(f"Saved new best model → {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion)
    print(f"Test accuracy: {test_acc:.4f}")
    plot_confusion(preds, labels, classes, os.path.join(DATA_ROOT, "text_mor_confusion.png"))

if __name__ == "__main__":
    main()
