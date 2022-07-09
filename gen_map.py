import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal

class BeliefSpace():
    OBSERVATION_CONFIDENCES = 0.95
    def __init__(self, xdim, ydim):
        x,y = np.meshgrid(np.arange(0,xdim,1), np.arange(0,ydim,1))
        self.xdim = xdim
        self.ydim = ydim
        self.pos = np.array([x,y]).transpose().reshape(xdim*ydim,2)
        us = self.gen_map_value(xdim, ydim, 0.04, DISPLAY=False) - 0.5
        vs = self.gen_map_value(xdim, ydim, 0.04, DISPLAY=False) - 0.5
        # bs = self.gen_map_value(xdim, ydim, 0.1, DISPLAY=False)
        bs = np.zeros((xdim,ydim))
        self.img = np.dstack((us,vs,bs))

    @staticmethod
    def gen_map_value(out_xdim, out_ydim, sample_factor, DISPLAY=False):
        """
        creates a map of values from a sparse collection of values
        sample_factor is the proportion of points to act as the constraint
        1 - sample_factor is the proportion of points interpolated
        """
        xdim = int(out_xdim * sample_factor)
        ydim = int(out_ydim * sample_factor)
        incr = sample_factor
        values = np.random.rand((xdim + 1) * (ydim + 1))
        xys = np.zeros(((xdim + 1) * (ydim + 1), 2))
        new_xys = np.zeros((out_xdim * out_ydim, 2))

        for i in range(xdim + 1):
            for j in range(ydim + 1):
                xys[i * (ydim + 1) + j] = (i, j)
        for i in range(int(xdim / incr)):
            for j in range(int(ydim / incr)):
                new_xys[int(i * ydim / incr + j)] = (i * incr, j * incr)
        interps = griddata(xys, values, new_xys, method="cubic")

        res_map = np.zeros((out_xdim, out_ydim))
        for e, (i, j) in enumerate(new_xys):
            # should be really close to a full integer
            res_map[int(round(i / sample_factor))][int(round(j / sample_factor))] = interps[e]

        if DISPLAY:
            ax = plt.axes(projection="3d")
            ax.scatter3D(new_xys.T[0], new_xys.T[1], interps, c=interps, cmap="Greens")
            plt.show()
        return res_map

    def info_gain(self,x,y):
        '''
        take a measurment at point (x,y) and record the resulting change in belief map
        '''
        u, v =  self.img[x][y][:-1]
        try:
            rv = multivariate_normal(
                [x, y],
                [
                    [20*abs(u), 0],
                    [0, 20*abs(v)],
                ],
            )
        except np.linalg.LinAlgError:
            print(
                "ran into another singurity :("
            )  # did nothing wrong the math just freaks out
        pos_pdf = rv.pdf(self.pos)
        zero_prob = max(pos_pdf)
        confidences = pos_pdf * self.OBSERVATION_CONFIDENCES / zero_prob
        oldsum = np.sum(self.img)
        self.img[:,:,2] = np.maximum(self.img[:,:,2], confidences.reshape(self.xdim, self.ydim))
        return np.sum(self.img) - oldsum



def main():
    xdim = 50
    ydim = 100
    X,Y = np.meshgrid(np.arange(0,xdim,1), np.arange(0,ydim,1))
    bspace = BeliefSpace(xdim, ydim)
    xys = []
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
