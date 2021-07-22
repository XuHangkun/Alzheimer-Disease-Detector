from .baseline_dataset import BaselineDataset

def create_dataset(name,x,y):
    if name.lower() == "baseline":
        return BaselineDataset(x,y)
    else:
        return BaselineDataset(x,y)
