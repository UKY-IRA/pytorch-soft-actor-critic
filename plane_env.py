import gym
from scipy.interpolate import griddata
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Plane(gym.Env):
    # =========================classifying plane state and dynamics==============
    # state constants
    v = 2  # cells/s, shooting for constant v
    pitch = 0  # working in 2D space for now
    g = 9.81  # m/s
    dt = 0.2  # control step
    # functions to describe state-to-action relationships
    action_max = 2*math.pi/9
    action_space = gym.spaces.Box(low=np.array([-action_max]), high=np.array([action_max]))
    xdim, ydim = (50, 90)
    obs_state_len = 4
    padding = 10
    maxtime = 100
    # observation space is a map of xdim x ydim with a value associated with each cell
    observation_space = gym.spaces.Box(
        low=0,
        high=max([xdim, ydim, maxtime, 2 * math.pi]),
        shape=(xdim + 1, ydim)
        # I dont really like this but basically, top row is our observer state
    )
    _max_episode_steps = int(maxtime/dt)
    # ===========================================================================

    def __init__(self, initial_state=None):
        self.gen_gainmap()
        if initial_state:  # start with a predetermined or random init
            self.x = initial_state[0][0]
            self.y = initial_state[0][1]
            self.yaw = initial_state[0][2]
            self.t = 0
            self.image = initial_state[1:]
            self._set_state_vector()
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
        self._set_state_vector()

        # simple mapped value reward
        if (
            round(self.x) < self.xdim
            and round(self.y) < self.ydim
            and self.x > 0
            and self.y > 0
        ):
            reward = self.info_gain(int(round(self.x)), int(round(self.y)))
            done = self.t >= self.maxtime
            info = None
        else:
            reward = -1
            done = True
            info = "out of bounds"
        return (self.normed_state(), reward, done, info)

    def gen_gainmap(self):
        depth = 5
        current_layer = [[0,0,1.0]]
        grid = np.zeros((2*depth+1, 2*depth+1))
        grid[depth][depth] = 1.
        decay = 0.55
        for d in range(1, depth+1):
            new_layer = []
            for dx, dy in [(-1,0),(1,0),(0,1),(0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]:
                for x, y, z in current_layer:
                    if z*decay > grid[int(x+dx)+depth][int(y+dy)+depth]:
                        grid[int(x+dx)+depth][int(y+dy)+depth] = z*decay
                    new_layer.append([x+dx, y+dy, z*decay])
            current_layer = new_layer
        self.depth = depth # i wanted to do i locally first, sue me
        self.grid = grid

    def info_gain(self, xind, yind):
        xmin = max([xind-self.depth,0])
        xmax = min([xind+self.depth,self.xdim])
        ymin = max([yind-self.depth,0])
        ymax = min([yind+self.depth,self.ydim])
        # reframe the decay grid into the image then subtract that value
        value = self.dt*np.multiply(self.image[xmin:xmax,ymin:ymax], self.grid[self.depth-(xind-xmin):self.depth+(xmax-xind),
                                                           self.depth-(yind-ymin):self.depth+(ymax-yind)])
        self.image[xmin:xmax,ymin:ymax] = self.image[xmin:xmax, ymin:ymax] - value
        return np.sum(value)

    def normed_state(self):
        norm = np.zeros(self.state.shape)
        norm[0][0] = self.state[0][0] / self.xdim
        norm[0][1] = self.state[0][1] / self.ydim
        norm[0][2] = self.state[0][2] / (2 * math.pi)
        norm[0][3] = self.state[0][3] / self.maxtime
        norm[1:] = self.state[1:]
        return norm

    def reset(self):
        self.x = np.random.choice(
            np.array(range(self.xdim - self.padding * 2)) + self.padding
        )
        self.y = np.random.choice(
            np.array(range(self.ydim - self.padding * 2)) + self.padding
        )
        self.yaw = np.random.rand() * 2 * math.pi
        self.image = self.gen_map_value(self.xdim, self.ydim, 0.1)
        self.t = 0
        self._set_state_vector()
        return self.normed_state()

    def _set_state_vector(self):
        self.state = np.zeros((self.xdim + 1, self.ydim))
        self.state[0][0] = self.x
        self.state[0][1] = self.y
        self.state[0][2] = self.yaw
        self.state[0][3] = self.t
        self.state[1:] = self.image

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
        return np.clip(res_map, a_min=0, a_max=1)

    @classmethod
    def _xdot(cls, yaw, pitch):
        return cls.v * math.cos(yaw) * math.cos(pitch)

    @classmethod
    def _ydot(cls, yaw, pitch):
        return cls.v * math.sin(yaw) * math.cos(pitch)

    @classmethod
    def _yawdot(cls, roll):
        return cls.g / cls.v * math.tan(roll)
