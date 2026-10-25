import torch.nn as nn
import torch

def build_criterion(args):
    criterion = args.criterion.lower()

    if criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif criterion == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif criterion == "mse":
        criterion = nn.MSELoss()
    elif criterion == "in":
        return None
    else:
        raise NotImplementedError(f"criterion {criterion} is not implemented")

    return criterion
