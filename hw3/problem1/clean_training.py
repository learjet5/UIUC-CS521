# !pip install tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
train_dataset = datasets.CIFAR10('cifar10_data/', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        elif self.activation[:6] == '3prelu':
            act = tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation[:8] == '3psmooth':
            act = tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
        activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print('no input normalization')
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(n_cls, cuda=True, half_prec=False, activation='relu', fts_before_bn=False,
    normal='none'):
    #print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec,
        activation=activation, fts_before_bn=fts_before_bn, normal=normal)


# intialize the model
model = PreActResNet18(10, cuda=True, activation='softplus1').to(device)
model.eval()

import os
from collections import deque

# ----------------------------
# Utils: evaluate
# ----------------------------
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_correct / total, total_loss / total

# ----------------------------
# Train setup
# ----------------------------
epochs = 80
base_lr = 0.1
momentum = 0.9
weight_decay = 5e-4
eval_every = 1
early_stop_acc = 0.80

model = model.to(device)
ce_loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)

scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

best_acc = 0.0
best_state = None
os.makedirs("models", exist_ok=True)

recent_accs = deque(maxlen=5)

print("Start training...")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_seen = 0

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{epochs}") as tepoch:
        for x, y in tepoch:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = ce_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = x.size(0)
            n_seen += n
            running_loss += loss.item() * n
            running_acc += (logits.argmax(1) == y).float().sum().item()

            tepoch.set_postfix({
                "lr": scheduler.get_last_lr()[0],
                "loss": f"{running_loss / n_seen:.4f}",
                "acc": f"{running_acc / n_seen:.4f}"
            })

    scheduler.step()

    if epoch % eval_every == 0:
        test_acc, test_loss = evaluate(model, test_loader, device)
        recent_accs.append(test_acc)
        print(f"[Eval] Epoch {epoch}: test_acc={test_acc:.4f}, test_loss={test_loss:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(best_state, "models/clean_trained.pth")

        if test_acc >= early_stop_acc:
            print(f"Early stop: reached {early_stop_acc*100:.0f}%+ accuracy at epoch {epoch}.")
            break

print(f"Training done. Best test accuracy: {best_acc:.4f}")
print("Saved best model to: models/clean_trained.pth")