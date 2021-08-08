import numpy as np
def split_data(x,y,train_ratio=0.9,kfold=1):
    """Split the data into train and valid set randomly
    """
    np.random.seed(7)
    assert x.shape[0] == y.shape[0]
    assert kfold <= int(1./(1 - train_ratio)) and 0 < kfold
    random_ids = np.array(range(x.shape[0]))
    for i in range(10):
        np.random.shuffle(random_ids)
    items_num = int(x.shape[0] / int(1./(1-train_ratio)))
    valid_ids = random_ids[(kfold-1)*items_num:min(kfold*items_num,x.shape[0])]
    train_ids = np.array([idx for idx in random_ids if idx not in valid_ids])
    train_x = x[train_ids,:]
    train_y = y[train_ids]
    valid_x = x[valid_ids,:]
    valid_y = y[valid_ids]
    return train_x,train_y,valid_x,valid_y,train_ids,valid_ids

def test():
    x = np.random.rand(1000,2)
    y = np.random.rand(1000)
    valids = []
    for i in range(10):
        train_x,train_y,valid_x,valid_y,train_ids,valid_id = split_data(x,y,kfold = i + 1)
        print(train_x.shape[0])
        print(valid_x.shape[0])
        valids += list(valid_id)
    valids = list(set(valids))
    valids = np.array(valids)
    print(valids.shape)


if __name__ == "__main__":
    test()
