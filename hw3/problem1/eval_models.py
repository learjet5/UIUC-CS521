# !pip install tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

# 
# Prepare dataset
# 
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
# batch_size = 64
batch_size = 128 # Here I choose to use larger size since my gpu is L40S

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
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))


# 
# Prepare ResNet model
# 
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

# Pre-activation Residual Block
# Normal: Input → Conv → BN → ReLU → Conv → BN → + shortcut → ReLU
# Pre-activation Block: Input → BN → ReLU → Conv → BN → ReLU → Conv → + shortcut
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

# Pre-activation ResNet
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
            print('PreActResNet: no input normalization')
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

# 
# Record standard accuracy
# 
def test_model_trainset_accuracy(model):
    model.train()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        with torch.no_grad():
            out = model(x_batch)
            pred = torch.argmax(out, dim=1)
            tot_acc += (pred == y_batch).sum().item()
            tot_test += y_batch.size(0)
    print('Train Set Standard accuracy %.5lf' % (tot_acc/tot_test))

def test_model_standard_accuracy(model):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        with torch.no_grad():
            out = model(x_batch)
            pred = torch.argmax(out, dim=1)
            tot_acc += (pred == y_batch).sum().item()
            tot_test += y_batch.size(0)
    print('Test Set Standard accuracy %.5lf' % (tot_acc/tot_test))

# 
# PGD Attack
# 

# Non-targeted attack requires the ground-truth label, which is also what the labels mean here.
# Delta is the perturbation; it should be similar to what eta is in FGSM.
# k: you can use K=10. Also, it would be fine to play with other values too
def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        # DONE: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        # DONE: compute the adv_x
        # find delta, clamp with eps
        with torch.no_grad():
            # update adv_x
            adv_x = adv_x + adv_x.grad.sign() * eps_step
            # projection to the eps-size ball around the initial starting input.
            delta = torch.clamp(adv_x - x, min=-eps, max=eps)
            adv_x = torch.clamp(x + delta, 0.0, 1.0)
        adv_x = adv_x.detach()
    return adv_x

def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]
        # DONE: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        # DONE: compute the adv_x
        # find delta, clamp with eps, project delta to the l2 ball
        # HINT: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
        with torch.no_grad():
            # update adv_x
            grad_view = grad.view(batch_size, -1)
            grad_norm = torch.norm(grad_view, p=2, dim=1, keepdim=True) + 1e-10
            normalized_grad = (grad_view / grad_norm).view_as(adv_x)
            adv_x = adv_x + eps_step * normalized_grad
            # projection to the eps-size ball around the initial starting input.
            delta = adv_x - x
            delta_view = delta.view(batch_size, -1)
            delta_norm = torch.norm(delta_view, p=2, dim=1, keepdim=True) + 1e-10
            factor = torch.minimum(torch.ones_like(delta_norm), eps / delta_norm)
            delta = (delta_view * factor).view_as(adv_x)
            adv_x = torch.clamp(x + delta, 0.0, 1.0)
        adv_x = adv_x.detach()
   
    return adv_x

# Added for HW3 problem1(b)
def fgsm_untargeted(model, x, labels, eps):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True)

    model.zero_grad()
    output = model(adv_x)
    loss = ce_loss(output, labels)
    loss.backward()

    with torch.no_grad():
        adv_x = adv_x + eps * adv_x.grad.sign()
        adv_x = torch.clamp(adv_x, 0.0, 1.0)

    return adv_x.detach()

# 
# Single-Norm Robust Accuracy Evaluation
# Now we only consider L_infinite attack.
# 
def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if attack == 'pgd_linf':
            # DONE: get x_adv untargeted pgd linf with eps, and eps_step=eps/4
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch, k=10, eps=eps, eps_step=eps/4.0)
        elif attack == 'pgd_l2':
            # DONE: get x_adv untargeted pgd l2 with eps, and eps_step=eps/4
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch, k=10, eps=eps, eps_step=eps/4.0)
        elif attack == 'fgsm':
            x_adv = fgsm_untargeted(model, x_batch, y_batch, eps=eps)
        else:
            x_adv = x_batch # no attack
        
        # get the testing accuracy and update tot_test and tot_acc
        with torch.no_grad():
            out = model(x_adv)
            pred = torch.argmax(out, dim=1)
            tot_acc += (pred == y_batch).sum().item()
            tot_test += y_batch.size(0)
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on {attack} attack with eps = {eps}')

# 
# Evaluate standard accuracy and robust accuracy.
# 
def test_a(model_name):
    assert os.path.exists(model_name), f"Model {model_name} does not exist. Please train it first."
    model.load_state_dict(torch.load(model_name, map_location=device))
    print("Model:", model_name, "under test a")
    test_model_standard_accuracy(model)
    test_model_on_single_attack(model, attack='pgd_linf', eps=8/255)
    print()

def test_b(model_name):
    assert os.path.exists(model_name), f"Model {model_name} does not exist. Please train it first."
    model.load_state_dict(torch.load(model_name, map_location=device))
    print("Model:", model_name, "under test b")
    test_model_on_single_attack(model, attack='fgsm', eps=4/255)
    test_model_on_single_attack(model, attack='fgsm', eps=8/255)
    test_model_on_single_attack(model, attack='fgsm', eps=12/255)
    test_model_on_single_attack(model, attack='fgsm', eps=16/255)
    print()

if __name__ == '__main__':
    # Clean model.
    ckpt = torch.load("models/clean_trained.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print("Cleanly trained model:")
    test_model_standard_accuracy(model)
    test_model_on_single_attack(model, attack='pgd_linf', eps=8/255)
    print()

    test_a("models/adv_trained_linf_0.00784313725490196.pth")
    test_a("models/adv_trained_linf_0.01568627450980392.pth")
    test_a("models/adv_trained_linf_0.03137254901960784.pth")
    test_a("models/adv_trained_linf_0.047058823529411764.pth")

    test_b("models/adv_trained_linf_0.00784313725490196.pth")
    test_b("models/adv_trained_linf_0.01568627450980392.pth")
    test_b("models/adv_trained_linf_0.03137254901960784.pth")
    test_b("models/adv_trained_linf_0.047058823529411764.pth")


