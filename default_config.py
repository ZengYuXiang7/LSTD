# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    logger: str = 'None'

@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 5
    epochs: int = 200
    patience: int = 20

    verbose: int = 0
    device: str = 'cuda'
    debug: bool = False
    experiment: bool = False
    program_test: bool = False
    record: bool = True


@dataclass
class BaseModelConfig:
    model: str = 'ours'
    rank: int = 40
    retrain: bool = True


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'rt'
    train_size: int = 500
    density: float = 0.80



@dataclass
class TrainingConfig:
    bs: int = 32
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'
    optim: str = 'AdamW'


@dataclass
class OtherConfig:
    classification: bool = False
    Ablation: int = 0
