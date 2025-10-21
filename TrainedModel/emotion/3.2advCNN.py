import os, random, copy, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# =============== 基础设置 ===============
def setup_font():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============== 数据与变换 ===============
def get_transforms(train=True, enhanced=False):
    if train:
        aug = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if enhanced:
            aug.insert(3, transforms.RandomRotation(20))
            aug.append(transforms.RandomErasing(p=0.15))
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class EmotionDataset(Dataset):
    def __init__(self, root, classes, transform=None, enhance_classes=None, mix_prob=0.15):
        self.root, self.classes = root, classes
        self.transform = transform
        self.enhance_classes = set(enhance_classes or [])
        self.mix_prob = mix_prob
        self.samples, self.targets = [], []
        for idx, c in enumerate(classes):
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append(os.path.join(cdir, f))
                    self.targets.append(idx)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx], self.targets[idx]
        img = Image.open(path).convert('RGB')
        c = self.classes[target]

        if c in self.enhance_classes:
            if random.random() < 0.5: img = img.filter(ImageFilter.GaussianBlur(random.uniform(0,0.7)))
            if random.random() < 0.5: img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8,1.2))

        img = self.transform(img)

        # MixUp（同类随机样本）
        if random.random() < self.mix_prob:
            j = random.randrange(len(self.samples))
            while self.targets[j] != target: j = random.randrange(len(self.samples))
            img2 = self.transform(Image.open(self.samples[j]).convert('RGB'))
            alpha = random.uniform(0.3,0.7)
            img = alpha * img + (1-alpha) * img2

        return img, target

def make_dataloaders(data_root, classes, batch=32, num_workers=4):
    enhance_classes = ["Confusion","Contempt","Disgust"]
    train_tf, val_tf = get_transforms(True,True), get_transforms(False)
    ds_train = EmotionDataset(os.path.join(data_root,"train"), classes, train_tf, enhance_classes)
    ds_val = EmotionDataset(os.path.join(data_root,"val"), classes, val_tf)
    ds_test = EmotionDataset(os.path.join(data_root,"test"), classes, val_tf)
    loader = lambda ds, shuf=True: DataLoader(ds,batch_size=batch,shuffle=shuf,num_workers=num_workers,pin_memory=True)
    return loader(ds_train), loader(ds_val,False), loader(ds_test,False)

# =============== 模型部分 ===============
class SpatialAttention(nn.Module):
    def __init__(self,k=7): super().__init__(); self.conv=nn.Conv2d(2,1,k,padding=3 if k==7 else 1,bias=False); self.sig=nn.Sigmoid()
    def forward(self,x): a=torch.mean(x,1,True); m,_=torch.max(x,1,True); att=self.sig(self.conv(torch.cat([a,m],1))); return x*att

class ChannelAttention(nn.Module):
    def __init__(self,c,r=16): super().__init__(); self.fc=nn.Sequential(nn.Conv2d(c,c//r,1),nn.ReLU(),nn.Conv2d(c//r,c,1)); self.sig=nn.Sigmoid()
    def forward(self,x): a=self.fc(F.adaptive_avg_pool2d(x,1)); m=self.fc(F.adaptive_max_pool2d(x,1)); return x*self.sig(a+m)

class CBAM(nn.Module):
    def __init__(self,c): super().__init__(); self.ca=ChannelAttention(c); self.sa=SpatialAttention(); 
    def forward(self,x): return self.sa(self.ca(x))

class EmotionNet(nn.Module):
    def __init__(self,num_classes,backbone='efficientnet_b3'):
        super().__init__()
        base=models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.features=base.features
        c=base.classifier[1].in_features
        self.cbam=CBAM(c)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Dropout(0.5), nn.Linear(c,1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512,num_classes)
        )
    def forward(self,x): f=self.cbam(self.features(x)); f=self.avg(f).flatten(1); return self.fc(f)

# =============== 损失函数 ===============
class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2.5): super().__init__(); self.a,self.g=alpha,gamma
    def forward(self,inp,tgt):
        ce=F.cross_entropy(inp,tgt,reduction='none',weight=self.a)
        pt=torch.exp(-ce); return ((1-pt)**self.g*ce).mean()

# =============== 训练逻辑 ===============
def train_epoch(model,loader,crit,opt,scaler,dev):
    model.train(); loss_sum, correct, total = 0,0,0
    for i,(x,y) in enumerate(tqdm(loader,desc='Train')):
        x,y=x.to(dev),y.to(dev)
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            out=model(x); loss=crit(out,y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update()
        loss_sum+=loss.item()*x.size(0); pred=out.argmax(1); correct+=pred.eq(y).sum().item(); total+=y.size(0)
        if i%100==0: torch.cuda.empty_cache()
    return loss_sum/total, correct/total

def validate(model,loader,crit,dev):
    model.eval(); loss_sum,correct,total=0,0,0; preds,tgts,probs=[],[],[]
    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for x,y in tqdm(loader,desc='Val'):
            x,y=x.to(dev),y.to(dev); out=model(x); loss=crit(out,y)
            p=F.softmax(out,1); loss_sum+=loss.item()*x.size(0); pred=out.argmax(1)
            correct+=pred.eq(y).sum().item(); total+=y.size(0)
            preds+=pred.cpu().tolist(); tgts+=y.cpu().tolist(); probs+=p.cpu().tolist()
    return loss_sum/total, correct/total, preds, tgts, probs

# =============== 可视化 ===============
def plot_confusion(tgts,preds,classes,save):
    cm=confusion_matrix(tgts,preds); cmn=cm.astype(float)/cm.sum(1)[:,None]
    plt.figure(figsize=(10,4))
    sns.heatmap(cmn,annot=True,fmt='.2f',xticklabels=classes,yticklabels=classes)
    plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Normalized Confusion'); plt.tight_layout()
    plt.savefig(os.path.join(save,'confusion.png')); plt.close()

# =============== 主程序 ===============
def main():
    setup_font(); set_seed()
    data_root=r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Preprocess"
    classes=sorted(os.listdir(os.path.join(data_root,'train')))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader,val_loader,test_loader=make_dataloaders(data_root,classes)
    model=EmotionNet(len(classes)).to(device)
    crit=FocalLoss(); opt=optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)
    scaler=GradScaler()
    best,best_w=0,None
    for epoch in range(20):
        tl,ta=train_epoch(model,train_loader,crit,opt,scaler,device)
        vl,va,_,_,_=validate(model,val_loader,crit,device)
        print(f"Epoch {epoch+1}: train_acc={ta:.3f}, val_acc={va:.3f}")
        if va>best: best, best_w=va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_w)
    torch.save(model.state_dict(), os.path.join(data_root,'efficient_cbam_clean.pth'))
    print("Model saved.")
    vl,va,preds,tgts,_=validate(model,test_loader,crit,device)
    print("Test acc:",va); plot_confusion(tgts,preds,classes,data_root)

if __name__ == '__main__':
    import multiprocessing; multiprocessing.freeze_support(); main()
