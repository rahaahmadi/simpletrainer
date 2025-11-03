import torch

def get_optimizer(name, model_params, lr=1e-4, weight_decay=1e-5, momentum=0.9):
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    
    elif name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"❌ Unsupported optimizer: {name}")
