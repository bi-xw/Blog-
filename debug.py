import torch
import random
import string
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def ptf_tensor(t,tag='tensor_',data_only=False):    
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 5))
    if (tag=='tensor_'):tag=tag+salt
    if data_only==False:
        print('\nThe info of {}:\n#############################\n  @dims: {}\n  @size: {}\n  @ele_sum: {}\n  @dtype: {}\n  @data:\n{}\n#############################\n'.format(tag,t.dim(),t.size(),t.numel(),t.dtype,t))
    else:
        print('\nThe info of {}:\n#############################\n  @data:\n{}\n#############################\n'.format(tag,t))


def draw3D_func(func,x_range=np.arange(-6, 6, 0.01),y_range=np.arange(-6, 6, 0.01)):
    X, Y = np.meshgrid(x_range, y_range) #生成网格点坐标矩阵
    Z=func([X,Y])
    ax = plt.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    plt.show()
