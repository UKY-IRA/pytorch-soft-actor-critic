from abc import ABC, abstractmethod 
from typing import List


class BeliefSpace():
    '''the space of information that is partially known about the map'''
    def __init__(self, dims: List[int]):
        assert len(dims) <= 3, "quit this extraplaner bullshit you're flying planes not spaceships! (belief space provided > 3 dimensions"

    @abstractmethod
    def get_window(self, point: List[int]):
        pass

    @abstractmethod
    def step(self, dt: float):
        pass
