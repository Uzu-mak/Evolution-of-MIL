import json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results'); OUT.mkdir(exist_ok=True)
print('device:', DEVICE, '| torch', torch.__version__)

TARGET = 9
MEAN_BAG, STD_BAG = 10, 2
TRAIN_BAGS = [50, 100, 200]      # paper sweeps 50-500
TEST_BAGS = 1000
SEEDS = [0, 1, 2]
EPOCHS = 20
LR, WEIGHT_DECAY = 5e-4, 1e-4
POOLINGS = ['mean', 'max', 'attention', 'gated_attention']

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST('data', train=True, download=True, transform=tfm)
mnist_test = datasets.MNIST('data', train=False, download=True, transform=tfm)

def to_tensors(ds):
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    return xs, ys

train_x, train_y = to_tensors(mnist_train)
test_x, test_y = to_tensors(mnist_test)

def make_bags(xs, ys, n_bags, seed):
    rng = np.random.default_rng(seed)
    bags = []
    for _ in range(n_bags):
        size = max(1, int(round(rng.normal(MEAN_BAG, STD_BAG))))
        idx = rng.integers(0, len(xs), size=size)
        inst = (ys[idx] == TARGET).float()
        bags.append((xs[idx], float(inst.max()), inst))
    return bags

test_bags = make_bags(test_x, test_y, TEST_BAGS, seed=12345)
print('test bags:', len(test_bags), '| positive share:', round(np.mean([b[1] for b in test_bags]), 3))

class MILNet(nn.Module):
    def __init__(self, pooling, L=500, D=128):
        super().__init__()
        self.pooling = pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.embed = nn.Sequential(nn.Linear(50 * 4 * 4, L), nn.ReLU())
        if pooling in ('attention', 'gated_attention'):
            self.V = nn.Linear(L, D)
            self.w = nn.Linear(D, 1)
            if pooling == 'gated_attention':
                self.U = nn.Linear(L, D)
        self.classifier = nn.Linear(L, 1)

    def forward(self, x):                      # x: (n, 1, 28, 28), one bag
        h = self.embed(self.features(x).flatten(1))         # (n, L)
        a = None
        if self.pooling == 'mean':
            z = h.mean(0)
        elif self.pooling == 'max':
            z = h.max(0).values
        else:
            s = torch.tanh(self.V(h))
            if self.pooling == 'gated_attention':
                s = s * torch.sigmoid(self.U(h))
            a = torch.softmax(self.w(s).squeeze(1), dim=0)    # (n,)
            z = (a.unsqueeze(1) * h).sum(0)
        return self.classifier(z).squeeze(), a

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def evaluate(model, bags):
    model.eval(); probs, labels, hits, positives = [], [], 0, 0
    with torch.no_grad():
        for x, y, inst in bags:
            logit, a = model(x.to(DEVICE))
            probs.append(torch.sigmoid(logit).item()); labels.append(y)
            if a is not None and y == 1:
                positives += 1
                hits += int(inst[a.argmax().item()].item() == 1)   # top-attended instance is a 9?
    preds = [int(p >= 0.5) for p in probs]
    out = {'auc': roc_auc_score(labels, probs), 'accuracy': accuracy_score(labels, preds)}
    if positives:
        out['attention_top1_is_target'] = hits / positives
    return out

def train_one(pooling, n_train, seed):
    set_seed(seed)
    bags = make_bags(train_x, train_y, n_train, seed=seed)
    model = MILNet(pooling).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        model.train(); random.shuffle(bags)
        for x, y, _inst in bags:
            logit, _ = model(x.to(DEVICE))
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y, device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    return model, evaluate(model, test_bags)

runs, start, best = [], time.time(), {}
for n_train in TRAIN_BAGS:
    for pooling in POOLINGS:
        for seed in SEEDS:
            model, metrics = train_one(pooling, n_train, seed)
            runs.append({'pooling': pooling, 'train_bags': n_train, 'seed': seed, **metrics})
            if pooling == 'gated_attention' and n_train == max(TRAIN_BAGS) and seed == SEEDS[0]:
                best['model'] = model
            print(f"{pooling:16s} train_bags={n_train:3d} seed={seed}  AUC={metrics['auc']:.3f}  acc={metrics['accuracy']:.3f}"
                  + (f"  top1_is_9={metrics['attention_top1_is_target']:.2f}" if 'attention_top1_is_target' in metrics else ''))
print(f'done in {(time.time() - start) / 60:.1f} min')

def summarize(runs):
    rows = {}
    for r in runs:
        rows.setdefault((r['pooling'], r['train_bags']), []).append(r)
    table = []
    for (pooling, n), rs in sorted(rows.items(), key=lambda kv: (kv[0][1], POOLINGS.index(kv[0][0]))):
        auc = np.array([r['auc'] for r in rs]); acc = np.array([r['accuracy'] for r in rs])
        top1 = [r['attention_top1_is_target'] for r in rs if 'attention_top1_is_target' in r]
        table.append({'pooling': pooling, 'train_bags': n, 'auc_mean': auc.mean(), 'auc_std': auc.std(),
                      'acc_mean': acc.mean(), 'acc_std': acc.std(),
                      'attention_top1_is_target': float(np.mean(top1)) if top1 else None})
    return table

table = summarize(runs)
lines = ['| Train bags | Pooling | Test AUC | Test accuracy | Top-attended instance is a 9 |', '| ---: | --- | ---: | ---: | ---: |']
for t in table:
    top1 = f"{t['attention_top1_is_target']:.1%}" if t['attention_top1_is_target'] is not None else '-'
    lines.append(f"| {t['train_bags']} | {t['pooling']} | {t['auc_mean']:.3f} ± {t['auc_std']:.3f} | {t['acc_mean']:.3f} ± {t['acc_std']:.3f} | {top1} |")
results_md = '\n'.join(lines)
print(results_md)

(OUT / 'results.json').write_text(json.dumps({'config': {'target': TARGET, 'mean_bag': MEAN_BAG, 'std_bag': STD_BAG, 'train_bags': TRAIN_BAGS,
    'test_bags': TEST_BAGS, 'seeds': SEEDS, 'epochs': EPOCHS, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'device': DEVICE,
    'torch': torch.__version__}, 'runs': runs, 'summary': table}, indent=2))
(OUT / 'results.md').write_text(results_md + '\n')

fig, ax = plt.subplots(figsize=(6, 4))
for pooling in POOLINGS:
    pts = [t for t in table if t['pooling'] == pooling]
    ax.errorbar([t['train_bags'] for t in pts], [t['auc_mean'] for t in pts], yerr=[t['auc_std'] for t in pts],
                marker='o', capsize=3, label=pooling)
ax.set_xlabel('Training bags'); ax.set_ylabel('Test AUC (mean ± sd over seeds)')
ax.set_title('MNIST-bags: pooling comparison'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'auc_vs_train_bags.png', dpi=150); plt.close(fig)

model = best['model']; model.eval()
x, y, inst = next(b for b in test_bags if b[1] == 1 and 6 <= len(b[0]) <= 12)
with torch.no_grad():
    _, a = model(x.to(DEVICE))
a = a.cpu().numpy()
fig, axes = plt.subplots(1, len(x), figsize=(1.3 * len(x), 1.9))
for i, axi in enumerate(axes):
    axi.imshow(x[i, 0] * 0.3081 + 0.1307, cmap='gray'); axi.axis('off')
    axi.set_title(f'{a[i]:.2f}', color='crimson' if inst[i] == 1 else 'black', fontsize=10)
fig.suptitle('Gated attention weight per digit (red = a 9)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'attention_example.png', dpi=150); plt.close(fig)
