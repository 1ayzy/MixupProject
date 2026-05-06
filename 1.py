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


# -------- mixup utils --------
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


# -------- test helper --------
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


# -------- train functions --------
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


# -------- experiment runner --------
def run_experiment(use_mixup, device, trainloader, testloader, alpha=1.0):
    model = PreActResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    name = "mixup" if use_mixup else "ERM"
    acc_history = []
    epoch_times = []

    for epoch in range(1, 201):
        start = time.time()
        if use_mixup:
            train_loss = train_mixup(model, trainloader, optimizer, criterion, alpha, device)
        else:
            train_loss = train_erm(model, trainloader, optimizer, criterion, device)
        test_acc = test(model, testloader, device)
        scheduler.step()
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        acc_history.append(test_acc)
        print(f"[{name}] Epoch {epoch:03d}: Test Acc={test_acc:.2f}%, Time={epoch_time:.2f}s")

    last10 = acc_history[-10:]
    avg_last10 = np.mean(last10)
    std_last10 = np.std(last10)
    best_acc = max(acc_history)

    results = {
        "method": name,
        "alpha": alpha if use_mixup else None,
        "last10_mean": round(avg_last10, 2),
        "last10_std": round(std_last10, 2),
        "best_accuracy": round(best_acc, 2),
        "total_time_min": round(sum(epoch_times) / 60, 1)
    }
    return model, acc_history, results


if __name__ == '__main__':
    # -------- data --------
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  # set to 0 for Windows
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting ERM training...")
    model_erm, hist_erm, res_erm = run_experiment(use_mixup=False, device=device, trainloader=trainloader,
                                                  testloader=testloader)
    print("\nStarting mixup training (alpha=1)...")
    model_mixup, hist_mixup, res_mixup = run_experiment(use_mixup=True, device=device, trainloader=trainloader,
                                                        testloader=testloader, alpha=1.0)

    all_results = {
        "ERM": res_erm,
        "mixup_alpha1": res_mixup,
        "history_erm": hist_erm,
        "history_mixup": hist_mixup
    }
    with open("results_exp1.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nFinal results saved to results_exp1.json")
    print("ERM:", res_erm)
    print("mixup:", res_mixup)