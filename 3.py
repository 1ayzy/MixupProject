import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import json

# -------- model (PreActResNet-18) --------
class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])

# -------- mixup --------
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------- test --------
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# -------- train loops --------
def train_erm(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def train_mixup(model, loader, optimizer, criterion, alpha, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# -------- noisy label generation --------
def corrupt_labels(dataset, noise_ratio=0.2, num_classes=10, seed=42):
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_samples = len(targets)
    num_noisy = int(noise_ratio * num_samples)
    noisy_indices = np.random.choice(num_samples, size=num_noisy, replace=False)
    for idx in noisy_indices:
        original_label = targets[idx]
        new_label = np.random.choice([c for c in range(num_classes) if c != original_label])
        targets[idx] = new_label
    dataset.targets = list(targets)
    return dataset

if __name__ == '__main__':
    NUM_EPOCHS = 150        # 150 эпох на каждую модель
    NOISE_RATIO = 0.2
    ALPHA_MIXUP = 8.0

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # noisy training set
    trainset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainset_noisy = corrupt_labels(trainset_clean, noise_ratio=NOISE_RATIO)
    trainloader_noisy = DataLoader(trainset_noisy, batch_size=128, shuffle=True, num_workers=0)

    # -------- ERM on noisy labels --------
    print(f"\n=== ERM on {int(NOISE_RATIO*100)}% noisy labels ({NUM_EPOCHS} epochs) ===")
    model_erm = PreActResNet18().to(device)
    criterion_erm = nn.CrossEntropyLoss()
    optimizer_erm = optim.SGD(model_erm.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_erm = optim.lr_scheduler.MultiStepLR(optimizer_erm, milestones=[100, 150], gamma=0.1)
    acc_erm = []
    start_erm = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_erm(model_erm, trainloader_noisy, optimizer_erm, criterion_erm, device)
        test_acc = test(model_erm, testloader, device)
        scheduler_erm.step()
        acc_erm.append(test_acc)
        print(f"Epoch {epoch:03d}: Test Acc={test_acc:.2f}%")
    time_erm = (time.time() - start_erm) / 60.0
    best_erm = max(acc_erm)
    last10_erm = acc_erm[-10:] if len(acc_erm) >= 10 else acc_erm
    avg_erm = float(np.mean(last10_erm))
    std_erm = float(np.std(last10_erm))

    # -------- mixup on noisy labels --------
    print(f"\n=== mixup (alpha={ALPHA_MIXUP}) on {int(NOISE_RATIO*100)}% noisy labels ({NUM_EPOCHS} epochs) ===")
    model_mixup = PreActResNet18().to(device)
    criterion_mix = nn.CrossEntropyLoss()
    optimizer_mix = optim.SGD(model_mixup.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler_mix = optim.lr_scheduler.MultiStepLR(optimizer_mix, milestones=[100, 150], gamma=0.1)
    acc_mixup = []
    start_mix = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_mixup(model_mixup, trainloader_noisy, optimizer_mix, criterion_mix, ALPHA_MIXUP, device)
        test_acc = test(model_mixup, testloader, device)
        scheduler_mix.step()
        acc_mixup.append(test_acc)
        print(f"Epoch {epoch:03d}: Test Acc={test_acc:.2f}%")
    time_mix = (time.time() - start_mix) / 60.0
    best_mix = max(acc_mixup)
    last10_mix = acc_mixup[-10:] if len(acc_mixup) >= 10 else acc_mixup
    avg_mix = float(np.mean(last10_mix))
    std_mix = float(np.std(last10_mix))

    results = {
        "noise_ratio": NOISE_RATIO,
        "epochs": NUM_EPOCHS,
        "ERM": {
            "best_accuracy": best_erm,
            "last10_mean": round(avg_erm, 2),
            "last10_std": round(std_erm, 2),
            "total_time_min": round(time_erm, 1)
        },
        "mixup": {
            "alpha": ALPHA_MIXUP,
            "best_accuracy": best_mix,
            "last10_mean": round(avg_mix, 2),
            "last10_std": round(std_mix, 2),
            "total_time_min": round(time_mix, 1)
        },
        "history_erm": acc_erm,
        "history_mixup": acc_mixup
    }
    with open("results_exp3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_exp3.json")
    print("ERM:", results["ERM"])
    print("mixup:", results["mixup"])