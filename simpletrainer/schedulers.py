import torch

def get_scheduler(name, optimizer, **kwargs):
    name = name.lower()

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "max"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            verbose=kwargs.get("verbose", True),
            min_lr=kwargs.get("min_lr", 1e-6)
        )

    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-6)
        )

    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1)
        )

    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95)
        )

    else:
        raise ValueError(f"❌ Unsupported scheduler: {name}")
