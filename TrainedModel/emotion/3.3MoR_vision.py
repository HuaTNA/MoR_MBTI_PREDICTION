"""
3.3MoR_Vision.py (optimized)
Vision emotion recognition with EfficientNet-B3 + MoR (refine) + CBAM.
- AMP + TF32
- Gradual unfreezing (freeze backbone for first 5 epochs)
- OneCycleLR scheduler
- Label smoothing CE
- Best checkpoint with metadata
- Confusion matrix export

Expected data layout:
<DATA_ROOT>/{train,val,test}/<ClassName>/*.jpg|png|jpeg
"""

import os, copy, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- Config -----------------------
DATA_ROOT = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Preprocess"
BATCH_SIZE = 32
EPOCHS = 15
FREEZE_EPOCHS = 5          # freeze backbone for first N epochs
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable TF32 for faster TensorCore math (Ampere+ GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------- Utils -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_classes(root_split):
    return sorted([d for d in os.listdir(root_split) if os.path.isdir(os.path.join(root_split, d))])

# ----------------------- Dataset -----------------------
class EmotionDataset(Dataset):
    """Simple image-folder dataset: one folder per class."""
    def __init__(self, root, classes, transform=None):
        self.samples, self.targets, self.classes = [], [], classes
        self.transform = transform
        for y, c in enumerate(classes):
            folder = os.path.join(root, c)
            if not os.path.isdir(folder): continue
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(folder, f))
                    self.targets.append(y)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.targets[i]

def make_loaders(data_root, classes):
    tf_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    mk = lambda split, tf: DataLoader(
        EmotionDataset(os.path.join(data_root, split), classes, tf),
        batch_size=BATCH_SIZE, shuffle=(split=="train"),
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return mk("train", tf_train), mk("val", tf_eval), mk("test", tf_eval)

# ----------------------- MoR + CBAM -----------------------
class MoRRouter2D(nn.Module):
    """
    Mixture-of-Recursions (MoR) refine block (depthwise conv + gate).
    Depth controls max # of refinement steps; we also allow early exit if gate is small.
    """
    def __init__(self, channels, depth=2, early_exit_thresh=0.2):
        super().__init__()
        self.depth = depth
        self.early_exit_thresh = early_exit_thresh
        # depthwise refine conv -> very lightweight
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.refine_bn = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for _ in range(self.depth):
            g = self.gate(x)                         # [B,1,1,1]
            update = self.refine_bn(self.refine(x))  # lightweight refine
            x = x + g * update
            # optional early-exit: if avg gate is tiny, stop refining
            if g.mean() < self.early_exit_thresh:
                break
        return x

class CBAM(nn.Module):
    """CBAM: Channel then Spatial attention."""
    def __init__(self, c):
        super().__init__()
        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c//16, c, 1), nn.Sigmoid()
        )
        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(torch.cat([x.mean(1,True), x.max(1,True)[0]], 1)) * x
        return x

class VisionMoRModel(nn.Module):
    """
    EfficientNet-B3 features -> MoR refine -> CBAM -> GAP -> Classifier.
    """
    def __init__(self, num_classes):
        super().__init__()
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.features = base.features
        c = base.classifier[1].in_features
        self.mor = MoRRouter2D(c, depth=2, early_exit_thresh=0.2)
        self.cbam = CBAM(c)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(c, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        f = self.mor(f)          # MoR first (stabilizes refinement)
        f = self.cbam(f)         # then attention
        f = self.avg(f).flatten(1)
        return self.head(f)

# ----------------------- Train / Eval -----------------------
def freeze_backbone(model, freeze=True):
    for p in model.features.parameters():
        p.requires_grad = not freeze

def make_optimizer_and_sched(model, steps_per_epoch):
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    # OneCycle over full run (we'll step every batch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=INIT_LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )
    return optimizer, scheduler

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train(); total=0; correct=0; loss_sum=0.0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if scheduler is not None: scheduler.step()

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        correct += logits.argmax(1).eq(y).sum().item()
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); total=0; correct=0; loss_sum=0.0; preds=[]; tgts=[]
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for x, y in tqdm(loader, desc="Eval"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs; loss_sum += loss.item() * bs
            p = logits.argmax(1)
            correct += p.eq(y).sum().item()
            preds += p.cpu().tolist(); tgts += y.cpu().tolist()
    return loss_sum/total, correct/total, preds, tgts

def save_confusion(tgts, preds, classes, save_path):
    cm = confusion_matrix(tgts, preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Normalized Confusion Matrix")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ----------------------- Main -----------------------
def main():
    set_seed(42)
    classes = list_classes(os.path.join(DATA_ROOT, "train"))
    print(f"Classes ({len(classes)}): {classes}")

    train_loader, val_loader, test_loader = make_loaders(DATA_ROOT, classes)
    model = VisionMoRModel(num_classes=len(classes)).to(DEVICE)

    # label smoothing CE (robust to noisy labels)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer, scheduler = make_optimizer_and_sched(model, steps_per_epoch=len(train_loader))
    scaler = GradScaler()

    # freeze backbone at the beginning
    freeze_backbone(model, freeze=True)

    best_acc, best_state = 0.0, None
    log_file = os.path.join(DATA_ROOT, "vision_mor_train_log.txt")

    for epoch in range(EPOCHS):
        # unfreeze after warmup
        if epoch == FREEZE_EPOCHS:
            freeze_backbone(model, freeze=False)
            print("Backbone unfrozen.")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"[{epoch+1:02d}/{EPOCHS}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{tr_acc:.6f}\t{va_acc:.6f}\n")

        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
            ckpt = {
                "state_dict": best_state,
                "classes": classes,
                "epoch": epoch + 1,
                "val_acc": best_acc,
                "arch": "EffNetB3+MoR+CBAM"
            }
            torch.save(ckpt, os.path.join(DATA_ROOT, "vision_mor_best.pth"))
            print(f"  ↳ Saved new best: val_acc={best_acc:.4f}")

    # Load best and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc, preds, tgts = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test accuracy: {te_acc:.4f}")

    # Save confusion matrix
    save_confusion(tgts, preds, classes, os.path.join(DATA_ROOT, "vision_mor_confusion.png"))
    print("Saved confusion matrix to vision_mor_confusion.png")

if __name__ == "__main__":
    main()
