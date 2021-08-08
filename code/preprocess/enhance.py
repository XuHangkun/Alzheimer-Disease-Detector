import numpy as np
import pandas as pd

def enhance_data(xs,ys,n_aug = 5):
    """
    enhance data
    """
    sm_xs,sm_ys = small_move(xs,ys,n_aug= n_aug)
    as_xs,as_ys = add_noise(xs,ys,n_aug= n_aug)
    new_xs = np.concatenate((sm_xs,as_xs,xs))
    new_ys = np.concatenate((sm_ys,as_ys,ys))
    return new_xs,new_ys

def small_move(xs,ys,n_aug=5,bias=0.01):
    """
    move according to other label
    """
    new_xs = []
    new_ys = []
    for x,y in zip(xs,ys):
        for i in range(n_aug):
            other_item_index = int(np.random.random()*len(xs))
            o_x = xs[other_item_index]
            ratio = abs(np.random.randn()*bias)
            new_x = x*(1-ratio) + ratio*o_x
            new_xs.append(new_x)
            new_ys.append(y)

    return np.array(new_xs),np.array(new_ys)

def add_noise(xs,ys,n_aug = 5,noise=0.01):
    """
    Add noise to each feature of each item
    """
    new_xs = []
    new_ys = []
    for x,y in zip(xs,ys):
        for i in range(n_aug):
            new_x = [xx*(1 + np.random.randn()*noise) for xx in x]
            new_xs.append(new_x)
            new_ys.append(y)
    return np.array(new_xs),np.array(new_ys)

def test():
    xs = np.array([[1,2,3,4,5],[2,1,4,6,8],[3,4,9,1,2],[2,8,2,9,5]])
    ys = np.array([1,0,1,0])
    new_xs,new_ys = enhance_data(xs,ys)
    print(new_xs)
    print(new_ys)

if __name__ == "__main__":
    test()
