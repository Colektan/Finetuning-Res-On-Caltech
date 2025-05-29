from collections import Counter
from config import *
import numpy as np
import os
from split_dataset import load_indices
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import Caltech101
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# 创建保存文件夹
model_save_path = os.path.join(SAVE_DIR, "best_model.pth")

# Tensorboard Writer
writer = SummaryWriter(log_dir=f'logs/{EXPERI_NAME}')

# 数据准备
dataset = Caltech101(root="./data")

train_indices = load_indices("./data/caltech101/train_indices.txt")
val_indices = load_indices("./data/caltech101/val_indices.txt")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5418, 0.5323, 0.5092], [0.3099, 0.3025, 0.3171])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5459, 0.5170, 0.4862], [0.3175, 0.3155, 0.3270])
])

class TransformApplier(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, target = self.subset[idx]
        data = data.convert("RGB")
        return self.transform(data), target

train_indices = load_indices("./data/caltech101/train_indices.txt")
val_indices = load_indices("./data/caltech101/val_indices.txt")

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_dataset = TransformApplier(train_dataset, train_transform)
val_dataset = TransformApplier(val_dataset, val_transform)

def create_balanced_sampler(dataset):
    """采样器：使训练集每个Epoch输出的样本数量均衡"""
    # 获取所有样本的标签
    if hasattr(dataset, 'targets'):  # 标准数据集格式
        labels = np.array(dataset.targets)
    else:  # 自定义数据集需要遍历
        labels = []
        for _, label in dataset:
            labels.append(label)
        labels = np.array(labels)
    
    # 统计每个类别的样本数
    class_counts = Counter(labels)
    print(f"原始类别分布: {dict(class_counts)}")
    
    # 计算每个样本的采样权重（类别倒数）
    class_weights = 1. / torch.Tensor(list(class_counts.values()))
    sample_weights = class_weights[labels]
    
    # 创建采样器（总样本数保持与原数据集一致）
    num_samples = len(labels)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True  # 允许重复采样小类别
    )
    return sampler

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 训练准备
model = resnet18(pretrained= not FROM_SCRATCH)
model.fc = nn.Linear(model.fc.in_features, 101) 
model = model.cuda()
loss_fn = nn.CrossEntropyLoss().cuda()

params_dict = [{'params': model.fc.parameters(), 'lr': FC_LR}]
other_params = []
for name, param in model.named_parameters():
    if 'fc' not in name:
        other_params.append(param)
params_dict.insert(0, {'params': other_params, 'lr': BASE_LR})

if OPTIMIZER == "SGD":
    optimizer = optim.SGD(params_dict, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "AdamW":
    optimizer = optim.AdamW(params_dict, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, GAMMA_EXP_SCHEDULER)

# 开始训练
iteration = 0
best_score = 0
for epoch in range(EPOCHES):
    model.train()
    running_loss = 0.0
    for d in tqdm(train_loader):
        inputs, labels = [t.cuda() for t in d]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/Train_iter', loss.item(), iteration)
        running_loss += loss.item() * inputs.size(0)
        iteration += 1
    
    epoch_loss = running_loss / len(train_dataset)
    writer.add_scalar('Loss/Train_epoch', epoch_loss, epoch + 1)
    print(f'Epoch {epoch+1}/{EPOCHES} Loss: {epoch_loss:.4f}')

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    epoch_loss = running_loss / len(val_dataset)
    writer.add_scalar('Acc/Eval_epoch', val_acc, epoch + 1)
    writer.add_scalar('Loss/Eval_epoch', epoch_loss, epoch + 1)
    print(f'Validation Accuracy: {val_acc:.2f}%')
    
    if val_acc > best_score:
        print(f"Best Acc has reached: {best_score} ---> {val_acc}")
        best_score = val_acc
        torch.save(model.state_dict(), model_save_path)
