import gym
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class BeliefSpace():
    OBSERVATION_CONFIDENCES = 0.95
    ENTROPY_PER_SECOND = 0.003
    def __init__(self, xdim, ydim):
        x,y = np.meshgrid(np.arange(0,xdim,1), np.arange(0,ydim,1))
        self.xdim = xdim
        self.ydim = ydim
        self.pos = np.array([x,y]).transpose().reshape(xdim*ydim,2)
        us = self.gen_map_value(xdim, ydim, 0.04, DISPLAY=False) - 0.5
        vs = self.gen_map_value(xdim, ydim, 0.04, DISPLAY=False) - 0.5
        bs = self.gen_map_value(xdim, ydim, 0.1, DISPLAY=False)
        # bs = np.zeros((xdim,ydim))
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

    def step(self, dt):
        self.img = self.img*(1-dt*self.ENTROPY_PER_SECOND)

class Plane(gym.Env):
    # =========================classifying plane state and dynamics==============
    # state constants
    v = 2  # cells/s, shooting for constant v
    pitch = 0  # working in 2D space for now
    g = 9.81  # m/s
    dt = 0.2  # control step
    action_max = 2*math.pi/9
    action_space = gym.spaces.Box(low=np.array([-action_max]), high=np.array([action_max]))
    xdim, ydim = (50, 100)
    obs_state_len = 4
    padding = 10
    maxtime = 100
    # observation space is a map of xdim x ydim with a value associated with each cell
    observation_space = gym.spaces.Box(
        low=0,
        high=max([1, maxtime, 2 * math.pi]),
        shape=(xdim + 1, ydim, 3)
        # I dont really like this but basically, top row is our observer state
    )
    _max_episode_steps = int(maxtime/dt)
    # ===========================================================================

    def __init__(self, initial_state=None, belief_space=None):
        if initial_state:  # start with a predetermined or random init
            self.x = initial_state[0][0][0]
            self.y = initial_state[0][0][1]
            self.yaw = initial_state[0][0][2]
            self.t = 0
            self.bspace = BeliefSpace(self.xdim, self.ydim)
            self.bspace.img = initial_state[1:]
            self._set_state_vector()
        elif belief_space:
            self.reset(belief_space=belief_space)
        else:
            self.reset()

    def step(self, action):
        if abs(action) >= self.action_max:
            print('illegal action')
            return (self.state, -1, True, "illegal action")  # illegal action
        control = action
        # changing the yaw first will give immediate feedback of the control
        self.yaw = (self.yaw + self._yawdot(control) * self.dt) % (2 * math.pi)
        self.x += self._xdot(self.yaw, self.pitch) * self.dt
        self.y += self._ydot(self.yaw, self.pitch) * self.dt
        self.t += self.dt

        # simple mapped value reward
        if (
            round(self.x) < self.xdim
            and round(self.y) < self.ydim
            and self.x > 0
            and self.y > 0
        ):
            reward = self.bspace.info_gain(int(round(self.x)), int(round(self.y)))
            # self.bspace.step(self.dt)
            done = self.t > self.maxtime
            info = None
        else:
            reward = -1
            done = True
            info = "out of bounds"
        self._set_state_vector()
        return (self.normed_state(), reward, done, info)

    def normed_state(self):
        norm = np.zeros(self.state.shape)
        norm[0][0][0] = self.state[0][0][0] / self.xdim
        norm[0][0][1] = self.state[0][0][1] / self.ydim
        norm[0][0][2] = self.state[0][0][2] / (2 * math.pi)
        norm[0][1][0] = self.state[0][1][0] / self.maxtime
        norm[1:] = self.state[1:]
        return norm

    def reset(self, belief_space=None):
        self.x = np.random.choice(
            np.array(range(self.xdim - self.padding * 2)) + self.padding
        )
        self.y = np.random.choice(
            np.array(range(self.ydim - self.padding * 2)) + self.padding
        )
        self.yaw = np.random.rand() * 2 * math.pi
        if belief_space:
            self.bspace = belief_space
        else:
            self.bspace = BeliefSpace(self.xdim, self.ydim)
        self.t = 0
        self._set_state_vector()
        return self.normed_state()

    def reset_state_from(self, state):
        self.x = state[0][0][0]
        self.y = state[0][0][1]
        self.yaw = state[0][0][2]
        self.t = state[0][1][0]
        self.bspace.img = state[1:]
        self._set_state_vector()

    def _set_state_vector(self):
        self.state = np.zeros((self.xdim + 1, self.ydim, 3))
        self.state[0][0][0] = self.x
        self.state[0][0][1] = self.y
        self.state[0][0][2] = self.yaw
        self.state[0][1][0] = self.t
        self.state[1:] = self.bspace.img

    @classmethod
    def _xdot(cls, yaw, pitch):
        return cls.v * math.cos(yaw) * math.cos(pitch)

    @classmethod
    def _ydot(cls, yaw, pitch):
        return cls.v * math.sin(yaw) * math.cos(pitch)

    @classmethod
    def _yawdot(cls, roll):
        return cls.g / cls.v * math.tan(roll)

if __name__ == '__main__':
    env = Plane()
    done = False
    X,Y = np.meshgrid(np.arange(0,Plane.xdim,1), np.arange(0,Plane.ydim,1))
    while not done:
        action = env.action_space.sample()
        s, r, done, info = env.step(action)
    plt.quiver(X.T,Y.T,env.bspace.img[:,:,0],env.bspace.img[:,:,1],env.bspace.img[:,:,2],cmap="RdYlGn")
    plt.show()
    print(info)
    print(s)
