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
# Schedules
# 
@dataclass
class Sched:
    kappa_start: float = 1.0
    kappa_end: float = 0.5
    eps_end: float = 0.1
    warmup_steps: int = 0
    ramp_steps: int = 20000
    def values(self, step: int):
        if step < self.warmup_steps:
            return float(self.kappa_start), 0.0
        t = min(1.0, (step - self.warmup_steps) / max(1, self.ramp_steps))
        kappa = self.kappa_start + t * (self.kappa_end - self.kappa_start)
        eps   = t * self.eps_end
        return float(kappa), float(eps)

# 
# IBP core (Conv/Linear + ReLU + Flatten/Pooling)
# 

def _affine_bounds_conv(xl, xu, conv: nn.Conv2d):
    W, b = conv.weight, conv.bias
    mu_prev = (xl + xu) / 2.0
    r_prev  = (xu - xl) / 2.0
    mu = F.conv2d(mu_prev, W, b, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    r  = F.conv2d(r_prev,  W.abs(), None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    return mu - r, mu + r

def _ensure_2d(x):
    return x if x.dim() == 2 else x.view(x.size(0), -1)

def _affine_bounds_linear(xl, xu, m: nn.Linear):
    # 保证输入是 (B, in_features)
    xl = _ensure_2d(xl)
    xu = _ensure_2d(xu)

    W = m.weight  # (out_features, in_features)
    b = m.bias

    # 你原来的 IBP 线性层上下界更新逻辑，例如：
    # 对于区间 [l, u]，线性层的上下界：
    # l' = l @ W^T + b - (W_neg @ (u - l)^T)^T 等等（取决于你写法）
    # 如果你是用 F.linear：
    l = torch.minimum(xl, xu)
    u = torch.maximum(xl, xu)
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)

    lb = F.linear(l, W_pos, b) + F.linear(u, W_neg, None)
    ub = F.linear(u, W_pos, b) + F.linear(l, W_neg, None)
    return lb, ub

def _relu_bounds(xl, xu):
    # standard interval ReLU
    return F.relu(xl), F.relu(xu)

def _flatten_bounds(xl, xu):
    return xl.view(xl.size(0), -1), xu.view(xu.size(0), -1)

def ibp_bounds(model: nn.Module, x: torch.Tensor, eps: float, elide_last: bool = True):
    xl = (x - eps).clamp(0.0, 1.0)
    xu = (x + eps).clamp(0.0, 1.0)

    last_linear = None
    modules = list(model.modules())

    # If eliding, skip the very last Linear and re-apply it with affine rule at the end
    if elide_last:
        # find last Linear in order
        for m in reversed(modules):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

    for m in modules:
        if m is model:  # skip container
            continue
        if elide_last and m is last_linear:
            break  # stop before last linear
        if isinstance(m, nn.Conv2d):
            xl, xu = _affine_bounds_conv(xl, xu, m)
        elif isinstance(m, nn.Linear):
            xl, xu = _affine_bounds_linear(xl, xu, m)
        elif isinstance(m, nn.ReLU):
            xl, xu = _relu_bounds(xl, xu)
        elif isinstance(m, nn.Flatten):
            xl, xu = _flatten_bounds(xl, xu)
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            xl, xu = m(xl), m(xu)
        else:
            # drop-in for layers without nonlinearity (no-op for BatchNorm here)
            pass

    if elide_last and last_linear is not None:
        xl, xu = _affine_bounds_linear(xl, xu, last_linear)

    return xl, xu  # logits bounds

def ibp_spec_ce_loss(logits, lb_logits, ub_logits, y, kappa: float):
    # Equation (12) loss: CE fit + CE spec
    ce_fit = F.cross_entropy(logits, y)
    z_hat = ub_logits.clone()
    z_hat.scatter_(1, y.view(-1,1), lb_logits.gather(1, y.view(-1,1)))
    ce_spec = F.cross_entropy(z_hat, y)
    return kappa * ce_fit + (1.0 - kappa) * ce_spec, ce_fit.item(), ce_spec.item()

def train_ibp(epochs=20, lr=1e-3, eps_train=0.1, warmup_steps=0, ramp_steps=20000, eps_eval=0.1, pgd_steps=50, pgd_restarts=5, out_dir="models"):
    opt = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)
    sched = Sched(kappa_end=0.5, eps_end=eps_train, warmup_steps=warmup_steps, ramp_steps=ramp_steps)

    global_step, best_nat, best_rob = 0, 0.0, 0.0

    for epoch in range(1, epochs+1):
        model.train()
        run, fit_run, spec_run, total = 0.0, 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            kappa, eps = sched.values(global_step)

            logits = model(x)
            lb, ub = ibp_bounds(model, x, eps, elide_last=True)
            loss, ce_fit_val, ce_spec_val = ibp_spec_ce_loss(logits, lb, ub, y, kappa)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = y.size(0)
            run += loss.item() * bs
            fit_run += ce_fit_val * bs
            spec_run += ce_spec_val * bs
            total += bs
            global_step += 1

            loop.set_postfix(kappa=f"{kappa:.3f}", eps=f"{eps:.3f}",
                             loss=f"{run/total:.4f}", fit=f"{fit_run/total:.4f}",
                             spec=f"{spec_run/total:.4f}")

        nat_loss, nat_acc = eval_nat(model, test_loader, device)
        # rob_acc = eval_robust_pgd(model, test_loader, device, eps=eps_eval, steps=pgd_steps, restarts=pgd_restarts)
        # print(f"Epoch {epoch:03d} | nat_acc={nat_acc:.4f} nat_loss={nat_loss:.4f} | rob_acc@PGD(eps={eps_eval})={rob_acc:.4f}")
        print(f"Epoch {epoch:03d} | nat_acc={nat_acc:.4f} nat_loss={nat_loss:.4f}")

        # if nat_acc > best_nat:
        #     best_nat = nat_acc
        #     torch.save(model.state_dict(), os.path.join(out_dir, "best_nat.pth"))
        # if rob_acc > best_rob:
        #     best_rob = rob_acc
        #     torch.save(model.state_dict(), os.path.join(out_dir, "best_rob.pth"))

    # Only save the last model
    torch.save(model.state_dict(), os.path.join(out_dir, "mnist_ibp.pth"))

# Invoke ibp model training
if not os.path.exists("models/"):
  os.mkdir("models")

start_time = time.time()
train_ibp(epochs=10)
end_time = time.time()
print(f"Total training time: {(end_time - start_time):.2f} seconds")

eps_eval=0.1; pgd_steps=50; pgd_restarts=5
nat_loss, nat_acc = eval_nat(model, test_loader, device)
rob_acc = eval_robust_pgd(model, test_loader, device, eps=eps_eval, steps=pgd_steps, restarts=pgd_restarts)
print(f"==> Done.", f"\nStandard accuracy: {nat_acc:.4f}", f"\nRobust accuracy (PGD @ eps={eps_eval}): {rob_acc:.4f}")

# Evaluate verified accuracy over a range of epsilons
# print_model_verified_accuracy("models/best_nat.pth")
# print_model_verified_accuracy("models/best_rob.pth")
print_model_verified_accuracy("models/mnist_ibp.pth")
