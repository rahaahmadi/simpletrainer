import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from simpletrainer.base_trainer import BaseTrainer
from simpletrainer.optimizers import get_optimizer
from simpletrainer.criteria import get_criterion

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64)

model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = get_criterion("bce")
optimizer = get_optimizer("adam", model.parameters(), lr=0.001)

trainer = BaseTrainer(model, train_loader, test_loader, criterion, optimizer, task_type="binary")
trainer.fit(num_epochs=10)
