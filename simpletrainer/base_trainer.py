import torch
import os
import wandb
import numpy as np
from simpletrainer.metrics import binary_metrics, multiclass_metrics, regression_metrics, multilabel_metrics
from simpletrainer.utils import save_model, load_model

class BaseTrainer:
    SUPPORTED_TASKS = ["binary", "multiclass", "multilabel", "regression"]
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, scheduler=None,
                 device=None, wandb_project=None, wandb_name=None, wandb_config=None, task_type="binary",
                 early_stopping_patience=None, grad_clip=None):
        
        if task_type not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_type '{task_type}'. Supported types are: {self.SUPPORTED_TASKS}")
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.early_stopping_patience = early_stopping_patience
        self.epochs_no_improve = 0
        self.grad_clip = grad_clip 
        
        if wandb_project:
            wandb.init(project=wandb_project, name=wandb_name,config=wandb_config, save_code=False)
            wandb.watch(self.model, log="all")
            
    def train(self):
        self.model.train()
        total_loss = 0
        all_labels, all_outputs = [], []

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * y.size(0)

            all_labels.extend(y.cpu().numpy())
            all_outputs.extend(output.detach().cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = self.compute_metrics(all_labels, all_outputs)
        metrics["loss"] = avg_loss
        return metrics

    def validate(self):
        self.model.eval()
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                all_labels.extend(y.cpu().numpy())
                all_outputs.extend(output.cpu().numpy())

        return self.compute_metrics(all_labels, all_outputs)

    def compute_metrics(self, labels, outputs, use_optimal_threshold=True):
        if self.task_type == "binary":
            return binary_metrics(labels, outputs, use_optimal_threshold)
        elif self.task_type == "multiclass":
            return multiclass_metrics(labels, outputs)
        elif self.task_type == "multilabel": 
            return multilabel_metrics(labels, outputs)
        elif self.task_type == "regression":
            return regression_metrics(labels, outputs)
        else:
            raise ValueError(f"Unsupported task_type '{self.task_type}'")
        
    def get_metric_key(self):
        if self.task_type == "binary":
            return "auc"
        elif self.task_type == "multiclass":
            return "f1"
        elif self.task_type == "multilabel":
            return "f1"
        elif self.task_type == "regression":
            return "loss"
    
    def fit(self, num_epochs, save_path=None):
        best_val_metric = -np.inf if self.task_type != "regression" else np.inf
        best_epoch = -1
        best_metrics = None

        for epoch in range(num_epochs):
            train_metrics = self.train()
            val_metrics = self.validate()
            
            key_metric = self.get_metric_key()
            key_metric_value = val_metrics[key_metric]
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(key_metric_value)
                else:
                    self.scheduler.step()

            is_best = (key_metric_value > best_val_metric) if self.task_type not in ["regression"] else (key_metric_value < best_val_metric)

            if is_best:
                best_val_metric = key_metric_value
                best_epoch = epoch
                best_metrics = val_metrics
                self._epochs_no_improve = 0
                if save_path:
                    self.save_model(f"{save_path}_best.pth", wandb_save=True)
            else:
                self.epochs_no_improve += 1
            
            if self.early_stopping_patience and self._epochs_no_improve >= self.early_stopping_patience:
                print(f"⚠️ Early stopping triggered at epoch {epoch}")
                break

            if save_path:
                self.save_model(f"{save_path}_epoch{epoch}.pth")

            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = epoch
            if wandb.run:
                wandb.log(log_dict)

            print(f"Epoch {epoch}: Train {train_metrics}, Val {val_metrics}")

        print(f"Best epoch: {best_epoch}, Best val metric: {best_val_metric:.4f}")
        return best_epoch, best_metrics


    def save_model(self, path):
        save_model(self.model, path)

    def load_model(self, path):
        self.model = load_model(self.model, path, device=self.device)
