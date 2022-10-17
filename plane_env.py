import math
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, entropy
import matplotlib.pyplot as plt
import gym

binary_entropy = lambda p_x: np.sum(-(p_x*np.log(p_x) + (1-p_x)*np.log(1-p_x)))

class BeliefSpace():
    OBSERVATION_CONFIDENCES = 0.95
    ENTROPY_PER_SECOND = 0.005 # TODO: maybe base this off of the change in gif?
    GAS_THRESH = 0.2
    def __init__(self, xdim, ydim):
        x, y = np.meshgrid(np.arange(0, xdim, 1), np.arange(0, ydim, 1))
        self.xdim = xdim
        self.ydim = ydim
        self.pos = np.array([x, y]).transpose().reshape(xdim*ydim, 2)
        us = np.zeros((xdim, ydim))
        vs = np.zeros((xdim, ydim))
        bs = np.full((xdim, ydim), 0.5)
        self.img = np.dstack((us, vs, bs))

    @staticmethod
    def farFromHalf(x,y,abs=abs):
        # returns the value that is the farthest from 0.5
        # note: locally binding the functions make them fast in vectorization
        return x if abs(x-0.5) > abs(y-0.5) else y

    def info_gain(self, z_x, z_y, z_u, z_v, z_g):
        '''
        z_x: x position of measurement
        z_y: y position of measurement
        z_u: u component measured
        z_v: v component measured
        z_g: gas measured
        '''
        # insert wind measurement into the map
        self.img[int(round(z_x))][int(round(z_y))][0] = z_u
        self.img[int(round(z_x))][int(round(z_y))][1] = z_v
        # create gas_mask from g,u,v scaled from 0.5 -> OBSERVATION_CONFIDENCES for entropy
        try:
            randvar = multivariate_normal(
                [z_x, z_y],
                [
                    [20*abs(z_u), 0],
                    [0, 20*abs(z_v)],
                ],
            )
        except np.linalg.LinAlgError:
            print(
                "ran into another singurity :("
            )  # did nothing wrong the math just freaks out
        pos_pdf = randvar.pdf(self.pos)
        zero_prob = max(pos_pdf)
        normed_pos_pdf = (pos_pdf * self.OBSERVATION_CONFIDENCES / zero_prob)
        normed_pos_pdf = np.maximum(normed_pos_pdf, np.full((5000,), 0.5))
        # - if gas is detected
        if z_g > self.GAS_THRESH: # value where measured gas is significant
            #   - extrapolate a positive gas detected forward from the current wind vector
            #   - i.e. gas will be going here
            confidences = normed_pos_pdf
        # - if gas is not detected
        else:
        #   - extrapolate a negative gas detected backward from the current wind vector
        #   - i.e. gas did not come from here
            confidences = 1 - normed_pos_pdf
        # apply the gas mask into the belief map, prioritizing least entropic result
        keep_lowest_entropy = np.vectorize(self.farFromHalf)
        oldent = binary_entropy(self.img[:, :, 2])
        self.img[:, :, 2] = keep_lowest_entropy(self.img[:,:,2], confidences.reshape(self.xdim, self.ydim))
        print('here')
        return binary_entropy(self.img[:, :, 2]) - oldent  # measure the entropy change between states and return it

    def get_window(self,x,y,window_radius):
        ''' get a slice of the current map based on a position'''
        xmin = max([0, x - window_radius])
        xmax = min([self.xdim - 1, x + window_radius])
        ymin = max([0, y - window_radius])
        ymax = min([self.ydim - 1, y + window_radius])
        window = np.full((2*window_radius,2*window_radius,3), -1.0)
        # broadcast our bounded image data onto its respective spot on the window
        # the window center is (x,y) so there is a linear tranformation where where (x,y) must be the center index
        window[window_radius-(x-xmin):window_radius+(xmax-x),window_radius-(y-ymin):window_radius+(ymax-y),:] = self.img[xmin:xmax,ymin:ymax,:]
        return window

    def step(self, dt):
        # step injects information decay into the system
        baseline = np.full((self.xdim, self.ydim), 0.5)
        self.img[:, :, 2] = (self.img[:, :, 2] - baseline)*(1-dt*self.ENTROPY_PER_SECOND) + baseline

class Plane(gym.Env):
    # =========================classifying plane state and dynamics==============
    # state constants
    v = 2  # cells/s, shooting for constant v
    pitch = 0  # working in 2D space for now
    g = 9.81  # m/s
    dt = 0.2  # control step
    maxtime = 99 # TODO ; max time should be related to number of frames in the recording, also time should not be given to the network state
    action_max = 2*math.pi/9
    action_space = gym.spaces.Box(low=np.array([-action_max]), high=np.array([action_max]))
    xdim, ydim = (50, 100)
    obs_state_len = 4
    padding = 10
    # observation space is a map of xdim x ydim with a value associated with each cell
    observation_space = gym.spaces.Box(
        low=0,
        high=max([1, maxtime, 2 * math.pi]),
        shape=(xdim + 1, ydim, 3)
        # I dont really like this but basically, top row is our observer state
    )
    _max_episode_steps = int(maxtime/dt)
    threshold = 0.05 # once the value of looking forward can at maximum be threshold % of immediate reward stop looking
    # ===========================================================================

    def __init__(self, animation, gamma, initial_state=None, belief_space=None):
        self.t = 0
        self.animation = animation
        self.window_radius = int(math.log(self.threshold, gamma)*self.v*self.dt)
        if initial_state:  # start with a predetermined or random init
            self.x = initial_state[0][0][0]
            self.y = initial_state[0][0][1]
            self.yaw = initial_state[0][0][2]
            self.bspace = BeliefSpace(self.xdim, self.ydim)
            self.bspace.img = initial_state[1:]
        elif belief_space:
            self.reset(belief_space=belief_space)
        else:
            self.reset()

    @property
    def map(self): # the map evolves with time based on the recorded animation
        return self.animation[self.timestep]

    @property
    def timestep(self):
        ''' returns how many steps the simulation has taken since the beginning of the sim '''
        return int(self.t/self.dt)

    @property
    def state(self):
        '''compiled array of all of the relevant state information'''
        temp = np.zeros((self.xdim + 1, self.ydim, 3))
        temp[0][0][0] = self.x
        temp[0][0][1] = self.y
        temp[0][0][2] = self.yaw
        temp[0][1][0] = self.t
        temp[1:] = self.bspace.get_window(int(round(self.x)),
                                          int(round(self.y)),
                                          self.window_radius) # NOTE: this is a POMDP state so we only partially know our map
        return temp

    @property
    def normed_state(self):
        '''normalized array of all of the relevant state information'''
        norm = np.zeros((self.xdim + 1, self.ydim, 3))
        norm[0][0][0] = self.x / self.xdim
        norm[0][0][1] = self.y / self.ydim
        norm[0][0][2] = self.yaw / (2 * math.pi)
        norm[0][1][0] = self.t / self.maxtime
        norm[1:] = self.bspace.get_window(int(round(self.x)),
                                          int(round(self.y)),
                                          self.window_radius) # NOTE: this is a POMDP state so we only partially know our map
        return norm

    def measure(self, x, y):
        '''take a measurement from the map (u, v, g) '''
        return self.map[x][y] # returns (u,v,g) from that point

    def step(self, action):
        if abs(action) >= self.action_max:
            print('illegal action')
            return (self.normed_state, -1, True, "illegal action")  # illegal action
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
            reward = self.bspace.info_gain(
                int(round(self.x)), # z_x
                int(round(self.y)), # z_y
                *self.measure(int(round(self.x)), int(round(self.y))) # z_u, z_v, z_g 
            )
            # TODO: map accuracy metric
            # map_difference = np.sum(np.abs((self.map > self.bspace.GAS_THRESH).astype(int) - self.bspace.img))
            # print(map_difference)
            # self.bspace.step(self.dt)
            done = self.t > self.maxtime
            info = None
        else:
            reward = -1
            done = True
            info = "out of bounds"
        return (self.normed_state, reward, done, info)

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
        return self.normed_state

    def reset_state_from(self, state):
        self.x = state[0][0][0]
        self.y = state[0][0][1]
        self.yaw = state[0][0][2]
        self.t = state[0][1][0]
        self.bspace.img = state[1:]

    @classmethod
    def _xdot(cls, yaw, pitch):
        return cls.v * math.cos(yaw) * math.cos(pitch)

    @classmethod
    def _ydot(cls, yaw, pitch):
        return cls.v * math.sin(yaw) * math.cos(pitch)

    @classmethod
    def _yawdot(cls, roll):
        return cls.g / cls.v * math.tan(roll)

if __name__ == "__main__":
    anim = np.load('gas_plume_recording.npy')
    env = Plane(anim)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
