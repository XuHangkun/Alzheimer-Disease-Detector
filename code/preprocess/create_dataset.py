from .baseline_dataset import BaselineDataset

def create_dataset(name,x,y,n_aug=0,tuning=0.015):
    if name.lower() == "baseline":
        dataset = BaselineDataset(x,y,n_aug=n_aug,tuning=tuning)
    else:
        dataset = BaselineDataset(x,y,n_aug=n_aug,tuning=tuning)
    print("Create dataset : n_aug : %d ; tuning : %.3f ; Num of data : %d !"%(dataset.n_aug,dataset.tuning,len(dataset)))
    return dataset
