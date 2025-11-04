# simpletrainer

## 🚀 Overview
**simpletrainer** keeps your PyTorch training tidy: metrics, early stopping, LR schedules, checkpoints, and optional W&B. Supports binary, multiclass, multilabel **classification** and **regression**.

---

## 🧩 Installation

> **Install name:** `torch-simpletrainer` • **Import name:** `simpletrainer`

### From PyPI
> Install the correct **PyTorch** build for your platform first
```bash

pip install torch-simpletrainer
```

### From source
```bash
git clone https://github.com/rahaahmadi/simpletrainer.git
cd simpletrainer
pip install -e .
```

---

## ⚙️ Quickstart Example

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from simpletrainer import BaseTrainer

X = torch.randn(512, 16)
y = (torch.randn(512) > 0).float()
ds = TensorDataset(X, y)
train_loader = DataLoader(ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(ds, batch_size=128)

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

trainer = BaseTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    task_type="binary",
    early_stopping_patience=5,
    grad_clip=1.0,
    wandb_project="my-project", # optional
)

best_epoch, best_metrics = trainer.fit(num_epochs=25, save_path="checkpoints/run/model")
print("Best epoch:", best_epoch)
print("Best metrics:", best_metrics)
```

---

## ✨ Features

- **Tasks:** `binary classification`, `multiclass classification`, `multilabel classification`, `regression`
- **Metrics:** AUC / F1 / Accuracy (classification), MSE / MAE / R² (regression)
- **Training features:** early stopping, gradient clipping, LR schedulers
- **Checkpointing:**  `save_model` / `load_model` utilities
- **Logging:** optional Weights & Biases (auto-skips if not installed)

---

## 🤝 Contributing

Issues and PRs are welcome.