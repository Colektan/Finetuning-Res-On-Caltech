import math
import random
from collections import defaultdict, Counter
from torch.utils.data import Subset
from torchvision.datasets import Caltech101

def load_indices(file_path):
    with open(file_path, "r") as f:
        indices_str = f.read().strip()
        indices = list(map(int, indices_str.split(',')))  # 按逗号分割并转为整数
    return indices

if __name__ == "__main__":
    data_root = "./data"
    random.seed(42) 
    val_percent = 0.15

    caltech_dataset = Caltech101(root=data_root, download=True)

    class_indices = defaultdict(list)
    for idx in range(len(caltech_dataset)):
        _, class_idx = caltech_dataset[idx]
        class_indices[class_idx].append(idx)

    val_indices = []
    train_indices = []

    for class_idx, indices in class_indices.items():
        n_class = len(indices)

        n_val = min(math.ceil(val_percent * n_class), 10)

        random.shuffle(indices)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_dataset = Subset(caltech_dataset, train_indices)
    val_dataset = Subset(caltech_dataset, val_indices)

    # 统计验证集中每个类别的样本数
    val_class_counts = Counter()
    for idx in val_indices:
        _, class_idx = caltech_dataset[idx]
        val_class_counts[class_idx] += 1

    # 打印验证集的类别分布
    print("\n验证集类别样本统计：")
    for class_idx, count in val_class_counts.items():
        class_name = caltech_dataset.categories[class_idx]
        print(f"{class_name}: {count} samples")

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    with open(r".\data\caltech101\train_indices.txt", "w") as f:
        f.write(",".join(map(str, train_indices)))

    with open(r".\data\caltech101\val_indices.txt", "w") as f:
        f.write(",".join(map(str, val_indices)))
