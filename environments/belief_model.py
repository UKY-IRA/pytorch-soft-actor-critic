from abc import ABC, abstractmethod 
import typing


class BeliefSpace():
    '''the space of information that is partially known about the map'''
    def __init__(self, dims: list[int]):
        assert len(dims) <= 3, "quit this extraplaner bullshit you're flying planes not spaceships! (belief space provided > 3 dimensions"

    @abstractmethod
    def get_window(self, point: list[int], window_radius: int):
        pass

    @abstractmethod
    def step(self, dt: float):
        pass
