from split_dataset import load_indices
import os
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import Caltech101
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

TEST_MODEL = "LR-Test_9"
BATCH_SIZE = 32
model_save_path = os.path.join("logs", TEST_MODEL, "best_model.pth")

# 数据准备
dataset = Caltech101(root="./data")
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

val_indices = load_indices("./data/caltech101/val_indices.txt")
val_dataset = Subset(dataset, val_indices)
val_dataset = TransformApplier(val_dataset, val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 训练准备
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 101) 
model.load_state_dict(torch.load(model_save_path))
model = model.cuda()
loss_fn = nn.CrossEntropyLoss().cuda()

model.eval()
correct = 0
total = 0
running_loss = 0.0
class_correct = torch.zeros(101)
class_total = torch.zeros(101)
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
print(f'Validation Accuracy: {val_acc:.2f}%')

