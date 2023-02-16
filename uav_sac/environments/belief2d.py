'''

a 2D belief mapping helper class to conceptualize the interactions of the agent with the current understanding of the gas map

for DART developers:  the Belief2D.img and Belief2D.saved_img will be the point of interaction of the entire algoritm with the flow field model,
look for TODOs for context

'''
from uav_sac.environments.belief_model import BeliefSpace
import numpy as np
from typing import Tuple
import random
import time
from scipy.ndimage import map_coordinates
from gdm.gmrf import GMRF_Gas_Wind_Efficient
from gdm.common.observation import Observation
from gdm.common.obstacle_map import ObstacleMap


class Belief2D(BeliefSpace):
    OBSERVATION_CONFIDENCES = 0.95
    ENTROPY_PER_SECOND = 0.005
    GAS_THRESH = 0.2

    def __init__(self, xdim, ydim, window_radius):
        super().__init__([xdim, ydim])
        self.xdim = xdim
        self.ydim = ydim
        self.window_radius = window_radius
        obstacles = ObstacleMap(2, (xdim, ydim))
        resolution = 2.
        self.wind_and_gas_map = GMRF_Gas_Wind_Efficient(obstacles, resolution=resolution)
        self.coordinate_space = np.meshgrid(np.linspace(0, int(ydim/resolution)-1, ydim), np.linspace(0, int(xdim/resolution)-1, xdim))
        self.entropy = None

    def measure(self, x, y):  # this will be much more accurate once estimation comes for WRF
        return self.img[x,y]

    @property
    def img(self):
        wind_est = self.wind_and_gas_map.getWindEstimate().toMatrix()
        return np.dstack((
            map_coordinates(np.flip(wind_est[0].T, axis=1), self.coordinate_space),
            map_coordinates(np.flip(wind_est[1].T, axis=1), self.coordinate_space),
            map_coordinates(np.flip(self.wind_and_gas_map.getGasEstimate().toMatrix().T, axis=1), self.coordinate_space)
        ))

    @img.setter
    def img(self):
        raise RuntimeError("Cannot set image directly!")
        

    def info_gain(self, measurement: Tuple):
        """
        z_x: x position of measurement
        z_y: y position of measurement
        z_u: u component measured
        z_v: v component measured
        z_g: gas measured
        z_t: time
        """
        assert len(measurement) == 6, "expected measurment: (x,y,u,v,g,t)"
        x, y, u, v, g, t = measurement
        obs = Observation(position=(x,y), gas=g, wind=(u,v), time=t, data_type="gas+wind")


        oldent = self.entropy
        
        self.wind_and_gas_map.addObservation(obs)
        self.entropy = np.sum(self.wind_and_gas_map.getGasUncertainty().toMatrix())

        if not oldent:
            return 0

        return (
            oldent - self.entropy
        )  # measure the entropy change between states and return it

    def get_window(self, point):
        """get a slice of the current map based on a position"""
        x, y = point
        xmin = max([0, x - self.window_radius])
        xmax = min([self.xdim - 1, x + self.window_radius])
        ymin = max([0, y - self.window_radius])
        ymax = min([self.ydim - 1, y + self.window_radius])
        window = np.full((2 * self.window_radius, 2 * self.window_radius, 3), -1.0)
        # broadcast our bounded image data onto its respective spot on the window
        # the window center is (x,y) so there is a linear tranformation where where (x,y) must be the center index
        window[
            self.window_radius - (x - xmin): self.window_radius + (xmax - x),
            self.window_radius - (y - ymin): self.window_radius + (ymax - y),
            :,
        ] = self.img[xmin:xmax, ymin:ymax, :]
        return window

def gdm_test():
    animaption = np.load("../../animations/452_plume.npy")
    frame = animaption[-1]
    xdim = frame.shape[0]
    ydim = frame.shape[1]
    belief = Belief2D(xdim, ydim, 15)
    dt = 0.1
    t = 0

    x = random.randint(0, xdim - 1) 
    y = random.randint(0, ydim - 1)
    u, v, z = frame[x,y]
    check = time.time()
    belief.info_gain((x,y,u,v,z,t))
    belief.img
    print(time.time()-check)
    t += dt


if __name__ == "__main__":
    gdm_test()
