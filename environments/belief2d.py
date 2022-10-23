from belief_model import BeliefSpace
from scipy.stats import multivariate_normal
import numpy as np


def binary_entropy(p_x):
    return np.sum(-(p_x * np.log(p_x) + (1 - p_x) * np.log(1 - p_x)))


class Belief2D(BeliefSpace):
    OBSERVATION_CONFIDENCES = 0.95
    ENTROPY_PER_SECOND = 0.005
    GAS_THRESH = 0.2

    def __init__(self, xdim, ydim, window_radius):
        super().__init__([xdim, ydim])
        x, y = np.meshgrid(np.arange(0, xdim, 1), np.arange(0, ydim, 1))
        self.xdim = xdim
        self.ydim = ydim
        self.window_radius = window_radius
        self.pos = np.array([x, y]).transpose().reshape(xdim * ydim, 2)
        us = np.zeros((xdim, ydim))
        vs = np.zeros((xdim, ydim))
        bs = np.full((xdim, ydim), 0.5)
        self.img = np.dstack((us, vs, bs))

    @staticmethod
    def farFromHalf(x, y, abs=abs):
        # returns the value that is the farthest from 0.5
        # note: locally binding the functions make them fast in vectorization
        return x if abs(x - 0.5) > abs(y - 0.5) else y

    def info_gain(self, z_x, z_y, z_u, z_v, z_g):
        """
        z_x: x position of measurement
        z_y: y position of measurement
        z_u: u component measured
        z_v: v component measured
        z_g: gas measured
        """
        # insert wind measurement into the map
        self.img[int(round(z_x))][int(round(z_y))][0] = z_u
        self.img[int(round(z_x))][int(round(z_y))][1] = z_v
        # create gas_mask from g,u,v scaled from 0.5 ->
        # OBSERVATION_CONFIDENCES for entropy
        try:
            randvar = multivariate_normal(
                [z_x, z_y],
                [
                    [20 * abs(z_u), 0],
                    [0, 20 * abs(z_v)],
                ],
            )
        except np.linalg.LinAlgError:
            print(
                "ran into another singurity :("
            )  # did nothing wrong the math just freaks out
        pos_pdf = randvar.pdf(self.pos)
        zero_prob = max(pos_pdf)
        normed_pos_pdf = pos_pdf * self.OBSERVATION_CONFIDENCES / zero_prob
        normed_pos_pdf = np.maximum(normed_pos_pdf, np.full((5000,), 0.5))
        if z_g > self.GAS_THRESH:  # - if gas is detected
            #   - extrapolate a positive gas detected forward from the current
            #     wind vector
            #   - i.e. gas will be going here
            confidences = normed_pos_pdf
        else:  # - if gas is not detected
            #   - extrapolate a negative gas detected backward from the current
            #     wind vector
            #   - i.e. gas did not come from here
            confidences = 1 - normed_pos_pdf
        # apply the gas mask into the belief map, prioritizing least entropic
        # result
        keep_lowest_entropy = np.vectorize(self.farFromHalf)
        oldent = binary_entropy(self.img[:, :, 2])
        self.img[:, :, 2] = keep_lowest_entropy(
            self.img[:, :, 2], confidences.reshape(self.xdim, self.ydim)
        )
        print("here")
        return (
            binary_entropy(self.img[:, :, 2]) - oldent
        )  # measure the entropy change between states and return it

    def get_window(self, x, y):
        """get a slice of the current map based on a position"""
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

    def step(self, dt):
        # step injects information decay into the system
        baseline = np.full((self.xdim, self.ydim), 0.5)
        self.img[:, :, 2] = (self.img[:, :, 2] - baseline) * (
            1 - dt * self.ENTROPY_PER_SECOND
        ) + baseline
