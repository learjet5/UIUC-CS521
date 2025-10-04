# !pip install tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

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

## Simple NN. You can change this if you want. If you change it, mention the architectural details in your report.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
        x = x.view((-1, 28*28))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1) # added softmax for probabilities
        return x

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), Net())

model = model.to(device)
model.train() # enable train mode

def train_model(model, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.3f}')

def test_model(model):
    model.eval() # enable eval mode
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy on images: {100 * correct / total}')

train_model(model, 15)
test_model(model)

## TODO: Write the interval analysis for the simple model
## you can use https://github.com/Zinoex/bound_propagation



#############################
# Interval Analysis (IBP) ###
#############################


# l=lower bound, u=upper bound
def interval_affine(l, u, W, b):
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    y_l = W_pos @ l + W_neg @ u + b
    y_u = W_pos @ u + W_neg @ l + b
    return y_l, y_u

def interval_relu(l, u):
    zl = torch.clamp(l, min=0.0)
    zu = torch.clamp(u, min=0.0)
    return zl, zu

def interval_normalize(l, u, mean=0.1307, std=0.3081):
    return (l - mean) / std, (u - mean) / std

# x is a concrete input, so the bounds are also concrete, we can easily propagate them.
# We propagate [x-eps, x+eps] through Normalize -> Linear -> ReLU -> Linear.
def get_logits_bounds(model, x, eps):
    l0 = torch.clamp(x - eps, 0.0, 1.0).view(-1)
    u0 = torch.clamp(x + eps, 0.0, 1.0).view(-1)

    l1, u1 = interval_normalize(l0, u0, mean=0.1307, std=0.3081)

    W1 = model[1].fc.weight.detach() # [200, 784]
    b1 = model[1].fc.bias.detach()   # [200]
    l2, u2 = interval_affine(l1, u1, W1, b1)

    l3, u3 = interval_relu(l2, u2)

    W2 = model[1].fc2.weight.detach() # [10, 200]
    b2 = model[1].fc2.bias.detach()   # [10]
    l4, u4 = interval_affine(l3, u3, W2, b2) # logits bounds

    return l4, u4  #shape: [10], [10]

@torch.no_grad()
def get_output_logits(model, x): # We can skip the softmax here.
    normalize = model[0] # Normalize()
    net = model[1]       # Net() with .fc and .fc2
    x = normalize(x)
    x = x.view(x.size(0), -1)
    x = F.relu(net.fc(x))
    logits = net.fc2(x)
    return logits

# Given eps, verify concrete sample x
@torch.no_grad()
def verify_sample(model, x, y_label, eps):
    logits = get_output_logits(model, x.unsqueeze(0)).squeeze(0)
    pred = int(torch.argmax(logits).item())
    correct = (pred == int(y_label.item()))

    l_logit, u_logit = get_logits_bounds(model, x.squeeze(0), eps)

    # certification condition: for all j != c, lower_bound(logit_c - logit_j) > 0
    if correct:
        c = pred
        margins = l_logit[c] - u_logit  # vector of size 10
        margins[c] = torch.tensor(float('inf'))  # we ignore j=c
        verified = torch.min(margins) > 0
    else:
        verified = False

    return bool(verified)

# Evaluate verified accuracy on the whole test set for 10 epsilons
@torch.no_grad()
def evaluate_verified_accuracy(model, test_loader, epsilons):
    model.eval()
    total = 0
    results = {eps: {'verified': 0, 'eligible': 0} for eps in epsilons}
    # eligible = number of samples that are correctly classified without perturbation

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch = images.size(0)
        total += batch

        logits = get_output_logits(model, images)
        preds = logits.argmax(dim=1)

        # per-sample verification for each eps
        for i in range(batch):
            x = images[i:i+1].cpu()
            y = labels[i:i+1].cpu()
            clean_ok = (preds[i].item() == y.item())
            for eps in epsilons:
                if clean_ok:
                    results[eps]['eligible'] += 1
                    v = verify_sample(model.cpu(), x, y, eps)
                    if v:
                        results[eps]['verified'] += 1

    verified_acc = {eps: (100.0 * results[eps]['verified'] / max(1, results[eps]['eligible']))
                    for eps in epsilons}
    return verified_acc, results

# Run evaluation
eps_list = np.linspace(0.01, 0.1, 10).tolist()
verified_acc, raw = evaluate_verified_accuracy(model, test_loader, eps_list)
print("[Verified Accuracy over eps]:")
for eps in eps_list:
    print(f"  eps={eps:.3f}: {verified_acc[eps]:.2f}%  "
          f"(verified {raw[eps]['verified']}/{raw[eps]['eligible']} eligible)")