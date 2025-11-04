import torch
import numpy as np
from simpletrainer.metrics import binary_metrics, multiclass_metrics, regression_metrics, multilabel_metrics
from simpletrainer.utils import save_model, load_model
from tqdm import tqdm

class BaseTrainer:
    SUPPORTED_TASKS = ["binary", "multiclass", "multilabel", "regression"]

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=None,
        wandb_project=None,
        wandb_name=None,
        wandb_config=None,
        task_type="binary",
        early_stopping_patience=None,
        grad_clip=None,
    ):
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
        self.best_metrics = None

        self.use_wandb = False
        if wandb_project:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(project=wandb_project, name=wandb_name, config=wandb_config, save_code=False)
                self.wandb.watch(self.model, log="all")
                self.use_wandb = True
            except ImportError:
                print("wandb not installed; continuing without it.")

    def train(self):
        self.model.train()
        total_loss = 0
        all_labels, all_outputs = [], []

        for x, y in tqdm(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)

            # --- minimal shape/dtype patches ---
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                # expects logits: (B,) or (B,C); labels float
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.reshape(-1)
                y = y.float()

            elif isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                # expects logits (B,C) and labels (B,) long
                if output.ndim == 1:
                    output = output.unsqueeze(1)  # (B,) -> (B,1)
                if output.shape[1] == 1:
                    # convert single logit to two-class logits correctly
                    output = torch.cat([-output, output], dim=1)

            elif isinstance(self.criterion, torch.nn.MSELoss):
                # expects (B,) for both; labels float
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.reshape(-1)
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y.reshape(-1)
                y = y.float()
            # --- end patches ---

            loss = self.criterion(output, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * y.size(0)

            all_labels.extend(y.detach().cpu().numpy())
            all_outputs.extend(output.detach().cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = self.compute_metrics(all_labels, all_outputs)
        metrics["loss"] = avg_loss
        return metrics

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for x, y in tqdm(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)

                # --- mirror minimal patches here ---
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.reshape(-1)
                    y = y.float()

                elif isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    if output.ndim == 1:
                        output = output.unsqueeze(1)
                    if output.shape[1] == 1:
                        output = torch.cat([-output, output], dim=1)

                elif isinstance(self.criterion, torch.nn.MSELoss):
                    if output.ndim == 2 and output.shape[1] == 1:
                        output = output.reshape(-1)
                    if y.ndim == 2 and y.shape[1] == 1:
                        y = y.reshape(-1)
                    y = y.float()
                # --- end patches ---

                loss = self.criterion(output, y)
                total_loss += loss.item() * y.size(0)
                all_labels.extend(y.detach().cpu().numpy())
                all_outputs.extend(output.detach().cpu().numpy())

        avg_loss = total_loss / len(self.test_loader.dataset)
        metrics = self.compute_metrics(all_labels, all_outputs)
        metrics["loss"] = avg_loss
        return metrics

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

        for epoch in range(num_epochs):
            train_metrics = self.train()
            val_metrics = self.validate()

            key_metric = self.get_metric_key()
            key_metric_value = val_metrics[key_metric]

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    mode = getattr(self.scheduler, "mode", "min")
                    monitor = key_metric_value
                    if self.task_type != "regression":
                        if mode == "min":
                            monitor = -monitor
                    else:
                        if mode == "max":
                            monitor = -monitor

                    self.scheduler.step(monitor)
                else:
                    self.scheduler.step()


            is_best = (key_metric_value > best_val_metric) if self.task_type != "regression" else (key_metric_value < best_val_metric)

            if is_best:
                best_val_metric = key_metric_value
                best_epoch = epoch
                self.best_metrics = val_metrics
                self.epochs_no_improve = 0
                if save_path:
                    self.save_model(f"{save_path}_best.pth")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs.")

            if self.early_stopping_patience and self.epochs_no_improve >= self.early_stopping_patience:
                print(f"⚠️ Early stopping triggered at epoch {epoch}")
                break

            if save_path:
                self.save_model(f"{save_path}_epoch{epoch}.pth")

            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = epoch
            if self.use_wandb:
                self.wandb.log(log_dict)

            print(f"Epoch {epoch}: Train {train_metrics}, Val {val_metrics}\n")

        print(f"Best epoch: {best_epoch}, Best val metric: {self.best_metrics}")
        return best_epoch, self.best_metrics

    def save_model(self, path):
        save_model(self.model, path)

    def load_model(self, path):
        self.model = load_model(self.model, path, device=self.device)
