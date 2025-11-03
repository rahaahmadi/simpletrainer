import torch.nn as nn

def get_criterion(name="cross_entropy", **kwargs):
    name = name.lower()
    
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=kwargs.get("weight", None))
    
    elif name == "mse":
        return nn.MSELoss()
    
    elif name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=kwargs.get("pos_weight", None))