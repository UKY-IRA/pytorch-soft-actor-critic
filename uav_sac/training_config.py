'''
Training config loader helper with some nice type helpers. Just a different,
dataclass-centric way to declutter argparsing.
'''
from dataclasses import dataclass
from typing import Annotated, Union, get_type_hints
import json


@dataclass
class ValueRange:
    min: Union[float, int]
    max: Union[float, int]
    help: str = ""
    def validate(self, val):
        return val >= self.min and val <= self.max


@dataclass
class Config:
    def __post_init__(self):
        hints = get_type_hints(self, include_extras=True)
        for attr, hint in hints.items():
            validators = getattr(hint, '__metadata__', [])
            for validator in validators:
                if validator:
                    check = validator.validate(getattr(self, attr))
                    fail_str = f"Invalid value for {attr}, {getattr(self, attr)}"
                    if help := getattr(validator, "help", None):
                        fail_str += f": {help}."
                    assert check, fail_str


@dataclass
class HyperparamsConfig(Config):
    gamma: Annotated[float, ValueRange(0, 1, "recursive value decay")]
    tau: Annotated[float, ValueRange(0, 1, "polyak average ratio")]
    lr: Annotated[float, ValueRange(0, 1, "AdaMax learning rate")]
    alpha: Annotated[float, ValueRange(0, 1, "exploration incentive coefficient")]
    automatic_entropy_tuning: bool
    hidden_size: int
    batch_size: int


@dataclass
class EpisodeConfig(Config):
    replay_size: int
    start_steps: int
    num_planes: int
    num_steps: int
    updates_per_step: Union[float, int]



@dataclass
class TrainingConfig(Config):
    env_name: str
    policy: str
    hyperparams: HyperparamsConfig
    episode: EpisodeConfig
    seed: int
    cuda: bool


def training_config_from_json(json_file: str) -> TrainingConfig:
    train_cfg = None
    with open(json_file, "r") as cfg_file:
        # TODO: we can automate it i know we can!
        cfg = json.load(cfg_file)
        hyp_cfg = HyperparamsConfig(**cfg["hyperparams"])
        cfg.pop("hyperparams")
        ep_cfg = EpisodeConfig(**cfg["episode"])
        cfg.pop("episode")
        train_cfg = TrainingConfig(**cfg, episode=ep_cfg, hyperparams=hyp_cfg)
    return train_cfg
    
if __name__ == '__main__':
    res = training_config_from_json(f"training_config.json")
    print(res)
    print(res.env_name)