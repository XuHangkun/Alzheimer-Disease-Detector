import numpy as np
def split_data(x,y,train_ratio=0.8):
    """Split the data into train and valid set randomly
    """
    assert x.shape[0] == y.shape[0]
    train_ids = []
    valid_ids = []
    for i in range(x.shape[0]):
        if np.random.random() < train_ratio:
            train_ids.append(i)
        else:
            valid_ids.append(i)
    train_x = x[train_ids,:]
    train_y = y[train_ids]
    valid_x = x[valid_ids,:]
    valid_y = y[valid_ids]
    return train_x,train_y,valid_x,valid_y