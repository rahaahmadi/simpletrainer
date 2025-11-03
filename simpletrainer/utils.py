import torch
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(model, path, device=None):
    print(f"🔄 Loading model from {path} to {device}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
