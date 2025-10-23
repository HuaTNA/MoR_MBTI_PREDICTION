"""
4.3MoR_Text_from_csv.py
Train MBTI text classifier DIRECTLY from mbti_1.csv (type, posts)
- No folder-of-txts required
- tqdm progress bars, GPU-first (AMP/TF32)
- Stratified train/val/test split in-memory
- Options: per-user vs per-post samples, min length filter, cap per class, cache dir, local model path

Example:
  python 4.3MoR_Text_from_csv.py --csv ".\\data\\Text\\mbti_1.csv" --per-post --min-chars 30 --train 0.8 --val 0.1 --test 0.1

Windows-friendly defaults:
  - num_workers=0 (avoids multiprocessing hang)
"""

import os, csv, re, time, copy, random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, logging as hf_logging
hf_logging.set_verbosity_error()

# ==================== Global Tweaks ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= Config =======================
CSV_DEFAULT = r"C:\Users\lnasl\Desktop\MBTInew\data\Text\mbti_1.csv"  # override with --csv
MODEL_NAME_DEFAULT = "prajjwal1/bert-tiny"  # change to bert-base-uncased for higher accuracy
MAX_LEN = 192
BATCH_SIZE = 32
NUM_EPOCHS = 6
LR = 3e-4
WD = 0.01
NUM_WORKERS = 0   # Windows-safe
SEED = 42
GRAD_CLIP = 1.0
LABEL_SMOOTH = 0.05
PARTIAL_UNFREEZE_EPOCHS = 2

MBTI_16 = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

# ====================== Utilities ======================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True

def clean_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("|||", "\n")  # split marker -> newlines
    s = URL_RE.sub("", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def read_csv_samples(csv_path: str, per_post: bool, min_chars: int) -> Tuple[List[Tuple[str,int]], List[str]]:
    """
    Returns (items, classes)
    items: list of (text, class_index)
    classes: sorted list of classes present (intersection with MBTI_16)
    """
    raw_rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        # Some CSVs may have stray BOM or weird header repeats; handle robustly
        data = f.read()
        # Normalize newlines
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        # Remove accidental duplicate header lines embedded in the middle (common in concatenations)
        lines = [ln for ln in data.split("\n") if ln.strip() != ""]
    # Re-create a CSV reader on the cleaned text
    import io
    f2 = io.StringIO("\n".join(lines))
    reader = csv.DictReader(f2)
    rows = []
    for r in reader:
        t = (r.get("type") or "").strip().upper()
        p = r.get("posts") or ""
        if not t or not p:
            continue
        if t.lower() == "type":  # in case header reappears
            continue
        rows.append((t, p))

    # Determine present classes
    present = sorted({t for (t, _) in rows if t in MBTI_16})
    if not present:
        raise RuntimeError("CSV 里没有检测到有效的 MBTI 类型列(type) 或内容(posts)。")
    cls2idx = {c:i for i,c in enumerate(present)}

    # Build items
    items: List[Tuple[str,int]] = []
    for t, posts in rows:
        if t not in cls2idx:
            continue
        if per_post:
            parts = [clean_text(x) for x in posts.split("|||")]
            for ptxt in parts:
                if len(ptxt) >= min_chars:
                    items.append((ptxt, cls2idx[t]))
        else:
            text = clean_text(posts.replace("|||", " "))
            if len(text) >= min_chars:
                items.append((text, cls2idx[t]))

    if not items:
        raise RuntimeError("过滤后没有样本。调小 --min-chars 或检查 CSV 格式（列名需为 type,posts）。")
    return items, present

def stratified_indices(items: List[Tuple[str,int]], ratios=(0.8,0.1,0.1), seed=SEED, max_per_class:int|None=None):
    """
    Split indices by label into train/val/test.
    Optionally cap per-class samples.
    """
    random.seed(seed)
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, (_, y) in enumerate(items):
        buckets[y].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in buckets.items():
        random.shuffle(idxs)
        if max_per_class is not None:
            idxs = idxs[:max_per_class]
        n = len(idxs)
        n_train = int(n*ratios[0]); n_val = int(n*ratios[1]); n_test = n - n_train - n_val
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:n_train+n_val]
        test_idx  += idxs[n_train+n_val:]
    random.shuffle(train_idx); random.shuffle(val_idx); random.shuffle(test_idx)
    return train_idx, val_idx, test_idx

# ======================== Dataset ======================
class CSVDataset(Dataset):
    def __init__(self, items: List[Tuple[str,int]], tokenizer, max_len=MAX_LEN):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        text, label = self.items[idx]
        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        item = {k:v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ========================= Model =======================
class ExpertBlock(nn.Module):
    """A lightweight expert used within the MoR router."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextMoRModel(nn.Module):
    """
    Text encoder with a Mixture-of-Recursions gating head.

    The module stacks several recursive layers. Each layer contains a mixture of
    experts and a learned depth gate. Given the current state `h_t`, we compute:

        experts_t = {E_i(h_t)}           # expert proposals
        w_t      = softmax(R_t(h_t))     # router over experts
        proposal = sum_i w_t[i] * experts_t[i]
        g_t      = sigmoid(D_t(h_t))     # depth gate
        h_{t+1}  = g_t * proposal + (1 - g_t) * h_t

    This allows the model to adaptively blend expert transformations across
    different recursion depths. Setting the number of recursions to 1 reduces the
    behaviour to a single MoE layer with residual gating.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        num_recursions: int = 3,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_recursions < 1:
            raise ValueError("num_recursions 必须 >= 1")
        if num_experts < 1:
            raise ValueError("num_experts 必须 >= 1")

        load_kwargs: Dict[str, object] = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            load_kwargs["local_files_only"] = True

        self.encoder = AutoModel.from_pretrained(model_name, **load_kwargs)
        hidden = self.encoder.config.hidden_size

        # Recursive mixture-of-experts blocks
        self.recursions = nn.ModuleList(
            [
                nn.ModuleList([ExpertBlock(hidden, dropout) for _ in range(num_experts)])
                for _ in range(num_recursions)
            ]
        )
        self.routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, num_experts),
                )
                for _ in range(num_recursions)
            ]
        )
        self.depth_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(num_recursions)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]

        for router, experts, gate_mlp in zip(self.routers, self.recursions, self.depth_gates):
            # Compute expert proposals
            expert_outputs = torch.stack([expert(h) for expert in experts], dim=1)
            weights = torch.softmax(router(h), dim=-1).unsqueeze(-1)
            proposal = torch.sum(weights * expert_outputs, dim=1)
            gate = torch.sigmoid(gate_mlp(h))
            h = gate * proposal + (1 - gate) * h

        return self.classifier(h)

# =================== Label Smoothing CE =================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, target):
        n = logits.size(-1)
        logp = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true = torch.zeros_like(logp)
            true.fill_(self.smoothing / (n - 1))
            true.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true * logp, dim=-1))

# ===================== Train / Eval =====================
def train_epoch(model, loader, criterion, optimizer, scaler, scheduler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(DEVICE, non_blocking=True) for k,v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()

        preds = logits.argmax(1)
        bs = batch["labels"].size(0)
        total += bs
        correct += preds.eq(batch["labels"]).sum().item()
        loss_sum += loss.item() * bs
    return loss_sum / max(1,total), correct / max(1,total)

@torch.no_grad()
def eval_epoch(model, loader, criterion, stage="Eval"):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, labels_all = [], []
    for batch in tqdm(loader, desc=stage):
        batch = {k: v.to(DEVICE, non_blocking=True) for k,v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
        preds = logits.argmax(1)
        bs = batch["labels"].size(0)
        total += bs
        correct += preds.eq(batch["labels"]).sum().item()
        loss_sum += loss.item() * bs
        preds_all += preds.cpu().tolist()
        labels_all += batch["labels"].cpu().tolist()
    return loss_sum / max(1,total), correct / max(1,total), np.array(preds_all), np.array(labels_all)

def plot_confusion(preds, labels, classes, out_png):
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    import seaborn as sns
    fig = plt.figure(figsize=(9,8))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(out_png); plt.close(fig)
    print(f"[Confusion] saved → {out_png}")

# ========================= Main =========================
def main():
    import argparse
    set_seed(SEED)

    parser = argparse.ArgumentParser(description="Train MBTI classifier directly from mbti_1.csv")
    parser.add_argument("--csv", type=str, default=CSV_DEFAULT, help="Path to mbti_1.csv")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME_DEFAULT, help="HF model name or local path")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (optional)")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer only from local cache")
    # sampling options
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--per-post", action="store_true", help="Treat each '|||' post as one sample")
    g.add_argument("--per-user", action="store_true", help="Treat the whole row as one sample")
    parser.add_argument("--min-chars", type=int, default=30, help="Drop samples shorter than this after cleaning")
    parser.add_argument("--max-per-class", type=int, default=0, help="Cap samples per class (0 = no cap)")
    # split
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val",   type=float, default=0.1)
    parser.add_argument("--test",  type=float, default=0.1)

    args = parser.parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[Device] Using {DEVICE}", flush=True)
    print(f"[CSV] {csv_path}", flush=True)

    # Tokenizer
    tok_kwargs = {"use_fast": True}
    if args.cache_dir: tok_kwargs["cache_dir"] = args.cache_dir
    if args.local_files_only: tok_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)

    # Build items
    per_post = args.per_post or (not args.per_user)  # default: per-post
    items, classes = read_csv_samples(str(csv_path), per_post=per_post, min_chars=args.min_chars)
    print(f"Classes ({len(classes)}): {classes}", flush=True)

    # Split
    max_per = args.max_per_class if args.max_per_class > 0 else None
    train_idx, val_idx, test_idx = stratified_indices(items, ratios=(args.train,args.val,args.test),
                                                      seed=SEED, max_per_class=max_per)
    print(f"Samples → train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}", flush=True)

    # Datasets & Loaders
    train_ds = CSVDataset(items, tokenizer, MAX_LEN)
    val_ds   = CSVDataset(items, tokenizer, MAX_LEN)
    test_ds  = CSVDataset(items, tokenizer, MAX_LEN)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = TextMoRModel(args.model_name, num_classes=len(classes), cache_dir=args.cache_dir, local_files_only=args.local_files_only).to(DEVICE)
    # gradual unfreeze
    for p in model.encoder.parameters(): p.requires_grad = False
    criterion = LabelSmoothingCE(LABEL_SMOOTH)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=='cuda'))

    best_acc, best_state = -1.0, None
    out_dir = csv_path.parent
    log_file = out_dir / "train_log_from_csv.tsv"
    with log_file.open("w", encoding="utf-8") as f:
        f.write("epoch\ttrain_acc\tval_acc\n")

    for epoch in range(NUM_EPOCHS):
        if epoch == 0:
            print(">>> Train classifier head only (encoder frozen)", flush=True)
        if epoch == 1:
            print(">>> Unfreeze last 4 transformer layers", flush=True)
            # unfreeze last 4
            try:
                enc = model.encoder.encoder
                layers = enc.layer
                for p in model.encoder.embeddings.parameters():
                    p.requires_grad = False
                for i in range(len(layers)-4, len(layers)):
                    for p in layers[i].parameters():
                        p.requires_grad = True
            except Exception:
                # tiny models may have different structures; if fails, just unfreeze all
                for p in model.encoder.parameters(): p.requires_grad = True
        if epoch == PARTIAL_UNFREEZE_EPOCHS:
            print(">>> Unfreeze ALL transformer layers", flush=True)
            for p in model.encoder.parameters(): p.requires_grad = True

        tl, ta = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler)
        vl, va, _, _ = eval_epoch(model, val_loader, criterion, stage="Eval")
        print(f"[{epoch+1:02d}] train_acc={ta:.4f} | val_acc={va:.4f}", flush=True)

        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{ta:.4f}\t{va:.4f}\n")

        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
            save_path = out_dir / "text_mor_from_csv_best.pth"
            torch.save({'model': best_state, 'classes': classes, 'epoch': epoch, 'val_acc': va}, save_path)
            print(f"Saved new best model → {save_path}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, stage="Test")
    print(f"Test accuracy: {test_acc:.4f}", flush=True)
    # Confusion matrix
    out_png = out_dir / "text_mor_from_csv_confusion.png"
    plot_confusion(preds, labels, classes, str(out_png))

    # Optional: print a short classification report head (macro stats)
    try:
        print(classification_report(labels, preds, target_names=classes, digits=3))
    except Exception:
        pass

if __name__ == "__main__":
    main()
