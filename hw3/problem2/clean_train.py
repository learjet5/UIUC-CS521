# !pip install tensorboardX

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)

## Dataloaders
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 
# Define required model with 3 hidden layers of 50 neurons each
# 
class MLP50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 50),    nn.ReLU(inplace=True),
            nn.Linear(50, 50),    nn.ReLU(inplace=True),
            nn.Linear(50, 10)  # outputs for 10 classes
        )

    def forward(self, x):
        # x: [B, 1, 28, 28] -> expand to [B, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)

model = MLP50().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss() 

#
# PGD evaluation
#
def pgd_linf(model, x, y, eps, alpha=None, steps=50, restarts=5):
    model.eval()
    if alpha is None:
        alpha = max(eps / 4.0, 1.0/255.0)
    best = x.clone()
    best_mis = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

    for _ in range(restarts):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        delta = (x + delta).clamp(0,1) - x
        delta.requires_grad_(True)
        for _ in range(steps):
            logits = model(x + delta)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            with torch.no_grad():
                delta.add_(alpha * delta.grad.sign()).clamp_(-eps, eps)
                delta[:] = (x + delta).clamp(0,1) - x
            delta.grad.zero_()
        with torch.no_grad():
            mis = (model(x + delta).argmax(1) != y)
            replace = mis & (~best_mis)
            best[replace] = (x + delta)[replace]
            best_mis |= mis
    return best

@torch.no_grad()
def eval_nat(model, loader, device):
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n

def eval_robust_pgd(model, test_loader, device, eps, steps, restarts, alpha=None):
    model.eval()
    if alpha is None:
        alpha = eps / 4.0
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            adv = pgd_linf(model, x, y, eps=eps, alpha=alpha, steps=steps, restarts=restarts)
        with torch.no_grad():
            logits = model(adv)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# 
# Interval Analysis
# 
def interval_affine(l: torch.Tensor, u: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    y_l = l @ W_pos.T + u @ W_neg.T + b
    y_u = u @ W_pos.T + l @ W_neg.T + b
    return y_l, y_u

def interval_relu(l: torch.Tensor, u: torch.Tensor):
    return torch.clamp(l, min=0.0), torch.clamp(u, min=0.0)

def interval_flatten(l: torch.Tensor, u: torch.Tensor):
    B = l.size(0)
    return l.view(B, -1), u.view(B, -1)

def interval_forward(model: nn.Module, l: torch.Tensor, u: torch.Tensor):
    def apply(m, l, u):
        if isinstance(m, nn.Sequential):
            for sub in m:
                l, u = apply(sub, l, u)
            return l, u

        elif isinstance(m, nn.Flatten):
            return interval_flatten(l, u)

        elif isinstance(m, nn.ReLU):
            # Relu propagation
            return torch.clamp(l, min=0.0), torch.clamp(u, min=0.0)

        elif isinstance(m, nn.Linear):
            # Fixed
            if l.dim() > 2:
                l, u = interval_flatten(l, u)

            in_features = m.weight.size(1)
            if l.size(-1) != in_features or u.size(-1) != in_features:
                l, u = interval_flatten(l, u)

            # linear propagation
            W = m.weight
            b = m.bias
            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)
            y_l = l @ W_pos.T + u @ W_neg.T + b
            y_u = u @ W_pos.T + l @ W_neg.T + b
            return y_l, y_u

        elif isinstance(m, nn.Identity):
            return l, u

        else:
            raise NotImplementedError(f"Unsupported module in interval_forward: {m.__class__.__name__}")

    if isinstance(model, nn.Sequential):
        return apply(model, l, u)
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        return apply(model.net, l, u)
    return apply(model, l, u)

def verified_by_margin_bounds(lb_logits: torch.Tensor, ub_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B, C = lb_logits.shape
    y = labels.view(-1, 1)
    lb_true = lb_logits.gather(1, y).squeeze(1)
    mask = torch.ones_like(ub_logits, dtype=torch.bool)
    mask.scatter_(1, y, False)
    neg_inf = torch.tensor(-1e9, device=ub_logits.device, dtype=ub_logits.dtype)
    ub_others = torch.where(mask, ub_logits, neg_inf)
    ub_max_other, _ = ub_others.max(dim=1)
    return (lb_true > ub_max_other)

@torch.no_grad()
def verify_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, device: torch.device) -> torch.Tensor:
    l = torch.clamp(x - eps, 0.0, 1.0).to(device)
    u = torch.clamp(x + eps, 0.0, 1.0).to(device)
    lb, ub = interval_forward(model, l, u)
    ok = verified_by_margin_bounds(lb, ub, y.to(device))
    return ok

@torch.no_grad()
def evaluate_verified_accuracy(model: nn.Module,
                               test_loader,
                               eps_list,
                               device: torch.device = torch.device("cpu"),
                               max_batches: int = None):
    model.eval()
    results = {float(eps): {"eligible": 0, "verified": 0} for eps in eps_list}
    for bidx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct_mask = (preds == y)
        if correct_mask.any():
            x_good = x[correct_mask]
            y_good = y[correct_mask]
            for eps in eps_list:
                eps = float(eps)
                results[eps]["eligible"] += x_good.size(0)
                ok = verify_batch(model, x_good, y_good, eps, device)
                results[eps]["verified"] += ok.sum().item()
        if (max_batches is not None) and (bidx + 1 >= max_batches):
            break
    out = {}
    for eps in eps_list:
        eps = float(eps)
        elig = max(1, results[eps]["eligible"])
        out[eps] = 100.0 * results[eps]["verified"] / elig
    return out, results

# Evaluate verified accuracy over a range of epsilons
def print_model_verified_accuracy(model_name):
    global device, test_loader
    model = MLP50().to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    eps_min = 0.01; eps_max = 0.1; eps_num = 10
    eps_list = [eps_min + i * (eps_max - eps_min) / (eps_num - 1) for i in range(eps_num)]
    va, raw = evaluate_verified_accuracy(model, test_loader, eps_list, device=device, max_batches=128)
    print(f"\n[{model_name}] Verified Accuracy over eps:")
    for eps in eps_list:
        e = float(eps)
        print(f"  eps={e:.3f}: {va[e]:.2f}%  (verified {raw[e]['verified']}/{raw[e]['eligible']} eligible)")

# 
# Standard training
# 
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                 # [B, 10]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

epochs = 10
best_test_acc = 0.0
if not os.path.exists("models/"):
  os.mkdir("models")

start_time = time.time()    
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc   = evaluate(model, test_loader, criterion, device, desc="Testing")

    print(f"Epoch {epoch:02d}/{epochs} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "models/mnist_standard_best.pth")
end_time = time.time()
print(f"Total training time: {(end_time - start_time):.2f} seconds")
print(f"Best test accuracy: {best_test_acc:.4f}")

rob_acc = eval_robust_pgd(model, test_loader, device, eps=0.1, steps=50, restarts=5)
print(f"Robust test accuracy: {rob_acc:.4f}")

print_model_verified_accuracy("models/mnist_standard_best.pth")