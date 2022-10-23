from abc import abstractmethod, ABC
import gym
import typing

class PlaneEnv(gym.Env, ABC):
    """
    An abstract class to ensure the implementation of all
    publically exposed points of a simulated plane environment.
    The inherited class will also need to abide by the requirements
    of gym.Env
    """

    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def measure(self, x: int, y: int) -> list[float]:
        """taking a sample from the map"""

    @abstractmethod
    @property
    def state(self):
        """
        compiled array of all of the relevant state information
        this should be the complete state, belief objects handle
        the partial state
        """

    @abstractmethod
    @property
    def normed_state(self):
        """normalized array of all of the relevant state information"""

    @abstractmethod
    @property
    def map(self):
        """
        accessor to the current map state however you would like
        to display that
        """
