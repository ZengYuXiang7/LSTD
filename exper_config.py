# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    rank: int = 100
    density: float = 0.10
    num_users: int = 142
    num_servs: int = 4500
    num_times: int = 64
    L_windows: int = 10
    S_windows: int = 10
    bs: int = 1024
    device: str = 'cuda'
    epochs: int = 300
    patience: int = 30

