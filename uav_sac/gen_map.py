import math
import random
import numpy as np
import matplotlib.pyplot as plt
from plane_env import BeliefSpace
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal

def main():
    xdim = 50
    ydim = 100
    X,Y = np.meshgrid(np.arange(0,xdim,1), np.arange(0,ydim,1))
    bspace = BeliefSpace(xdim, ydim)
    xys = []
    # window = bspace.get_window(45,23,10)
    for x in range(30):
        xys.append((x,x))

    plt.quiver(X.T,Y.T,bspace.img[:,:,0],bspace.img[:,:,1],bspace.img[:,:,2],cmap="RdYlGn")
    plt.show()
    r = 0
    for x,y in xys:
        r += bspace.info_gain(x,y)
    print(r)
    plt.quiver(X.T,Y.T,bspace.img[:,:,0],bspace.img[:,:,1],bspace.img[:,:,2],cmap="RdYlGn")
    plt.show()

if __name__ == '__main__':
    main()
