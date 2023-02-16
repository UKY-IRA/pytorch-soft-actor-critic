import math
import numpy as np
import gym
import uuid
from dataclasses import dataclass
# local imports
from uav_sac.training_config import TrainingConfig
from uav_sac.environments.uav_explorer import PlaneEnv
from uav_sac.environments.belief2d import Belief2D
from uav_sac.utils import load_random_animation

class Dubins2DUAV():
    v = 2
    pitch = 0
    g = 9.81  # m/s^2

    def __init__(self, x, y, yaw, dt):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.dt = dt

    def step(self, action):
        # changing the yaw first will give immediate feedback of the control
        self.yaw = (self.yaw + self._yawdot(action) * self.dt) % (2 * math.pi)
        self.x += self._xdot(self.yaw, self.pitch) * self.dt
        self.y += self._ydot(self.yaw, self.pitch) * self.dt

    @classmethod
    def _xdot(cls, yaw, pitch):
        return cls.v * math.cos(yaw) * math.cos(pitch)

    @classmethod
    def _ydot(cls, yaw, pitch):
        return cls.v * math.sin(yaw) * math.cos(pitch)

    @classmethod
    def _yawdot(cls, roll):
        return cls.g / cls.v * math.tan(roll)


@dataclass
class FullState():
    x: float
    y: float
    yaw: float
    t: float
    animation: np.array
    img: np.array


class Simple2DUAV(PlaneEnv):
    # =========================defining environment state=============
    # time
    dt = 0.2  # control step
    maxtime = 99  # TODO ; max time <-> recording frames
    _max_episode_steps = int(maxtime/dt)
    # action space
    action_max = 2*math.pi/9
    action_space = gym.spaces.Box(low=np.array([-action_max]), high=np.array([action_max]))

    # state space
    xdim, ydim = (50, 100)
    obs_state_len = 3
    padding = 10
    # observation_space = gym.spaces.Box(
    #     low=0,
    #     high=max([1, maxtime, 2 * math.pi]),
    #     shape=(xdim + 1, ydim, 3)
    # )
    threshold = 0.05  # reward depreciation tolerance for window
    # ===========================================================================

    def __init__(self, animation: np.ndarray,
                 cfg: TrainingConfig,
                 initial_state: np.ndarray = None, 
                 belief_space: Belief2D = None):
        '''
        all properties are derived from:
            self.animation -> np.array
            self.t -> int
            self.plane -> Dubins2D
            self.belief -> Belief2D
        '''
        self.uuid = int(uuid.uuid1())
        self.t = 15
        self.maxtime = int(animation.shape[0]*self.dt) - 1
        self.animation = animation
        window_radius = int(math.log(self.threshold, cfg.hyperparams.gamma)*Dubins2DUAV.v*self.dt)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max([1, 2*math.pi]),  # TODO: u, v max
            shape=(2*window_radius + 1, 2*window_radius, 3)
        )
        if initial_state:
            self.plane = Dubins2DUAV(initial_state[0][0][0],  # x
                                     initial_state[0][0][1],  # y
                                     initial_state[0][0][2],  # yaw
                                     self.dt)  
            self.belief = Belief2D(self.xdim, self.ydim, window_radius)
            self.belief.img = initial_state[1:]
        elif belief_space:
            self.reset(belief_space=belief_space, window_radius=window_radius)
        else:
            self.reset(window_radius=window_radius)

    @property
    def map(self):  # the map evolves with time based on the recorded animation
        return self.animation[self.timestep]

    def measure(self, x, y):
        ''' take a measurement from the map (u, v, g) '''
        return self.map[x][y]  # returns (u,v,g) from that point

    @property
    def timestep(self):
        '''return the current sim time in steps since start'''
        return int(self.t/self.dt)

    @property
    def state(self):
        '''partial observed state of the belief'''
        temp = np.zeros((self.belief.window_radius*2 + 1, self.belief.window_radius*2, 3))
        temp[0][0][0] = self.plane.x
        temp[0][0][1] = self.plane.y
        temp[0][0][2] = self.plane.yaw
        temp[1:] = self.belief.get_window([int(round(self.plane.x)),
                                          int(round(self.plane.y))])
        return temp

    @property
    def normed_state(self):
        '''normalized array of partial observed state'''
        norm = np.zeros((self.belief.window_radius*2 + 1, self.belief.window_radius*2, 3))
        norm[0][0][0] = self.plane.x / self.xdim
        norm[0][0][1] = self.plane.y / self.ydim
        norm[0][0][2] = self.plane.yaw / (2 * math.pi)
        norm[1:] = self.belief.get_window([int(round(self.plane.x)),
                                          int(round(self.plane.y))])
        return norm

    @property
    def full_state(self):
        '''complete state used for benchmarking and replay'''
        tmp = FullState(
            self.plane.x,
            self.plane.y,
            self.plane.yaw,
            self.t,
            self.animation,
        )
        return tmp

    @property
    def error(self):
        '''bigger number is worse'''
        return np.linalg.norm(self.map - self.belief.img)

    def step(self, action):
        if abs(action) >= self.action_max:
            print('illegal action')
            return (self.normed_state, -1_000, True, "illegal action")
        # give the plane its control action
        self.plane.step(action)
        self.t += self.dt

        # simple mapped value reward
        if self.check_bounds():
            x_ind = int(self.plane.x)
            y_ind = int(self.plane.y)
            measurement = self.measure(x_ind, y_ind)
            reward = self.belief.info_gain(
                (
                    x_ind,
                    y_ind,
                    measurement[0],
                    measurement[1],
                    measurement[2],
                    self.t
                )
            )
            # TODO: map accuracy metric
            # map_difference = np.sum(np.abs((self.map > self.belief.GAS_THRESH).astype(int) - self.belief.img))
            # print(map_difference)
            self.belief.step(self.dt)
            done = self.t > self.maxtime
            info = None
        else:
            reward = -1_000
            done = True
            info = "out of bounds"
        return (self.normed_state, reward, done, info)

    def check_bounds(self):
        '''check if plane is still in bounds of the map'''
        return ((self.plane.x < self.xdim and self.plane.x > 0) and \
               (self.plane.y < self.ydim and self.plane.y > 0))

    def reset(self, animation = None, belief_space = None, window_radius = None):
        window_radius = window_radius or self.belief.window_radius
        x = np.random.choice(
            np.array(range(self.xdim - self.padding * 2)) + self.padding
        )
        y = np.random.choice(
            np.array(range(self.ydim - self.padding * 2)) + self.padding
        )
        yaw = np.random.rand() * 2 * math.pi
        self.plane = Dubins2DUAV(x, y, yaw, self.dt)
        self.t = 0
        if animation is not None:
            self.animation = animation
            if belief_space:
                self.belief = belief_space
            else:
                self.belief = Belief2D(self.xdim, self.ydim, window_radius)
        else:
            # truth and estimation are coupled
            self.animation = load_random_animation()
            self.belief = Belief2D(self.xdim, self.ydim, window_radius)
            
        return self.normed_state

    ''' cant do it with new belief
    def reset_full_state_from(self, state: FullState):
        self.animation = state.animation
        self.belief.img = state.img
        self.plane = Dubins2DUAV(state.x, state.y, state.yaw, self.dt)
        self.t = state.t
    '''


if __name__ == "__main__":
    anim = np.load('../animations/gas_plume_recording.npy')
    env = Simple2DUAV(0.95, anim)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
