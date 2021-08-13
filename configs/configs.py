from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DAConfig:
    type: str = 'pda'


@dataclass
class SourceTrainConfig:
    worker: int = 4
    max_epoch: int = 50
    batch_size: int = 64
    lr: float = 1e-2
    seed: int = 2020
    smooth: float = 0.1
    epsilon: float = 1e-5


@dataclass
class ModelConfig:
    backbone: str = 'resnet50'
    layer: str = 'wn'
    classifier: str = 'bn'
    bottleneck_dim: int = 256


@dataclass
class OfficeHomeConfig(DAConfig):
    name: str = "office-home"
    root: str = "../../data"
    resize_size: int = 256
    crop_size: int = 224

    if DAConfig.type == "oda":
        num_class: int = 25
        src_class: int = 25
        tar_class: int = 65
    else:
        num_class: int = 65
        src_class: int = 65
        tar_class: int = 25

    s: int = 0
    name_src: str = MISSING
    name_tar: str = MISSING


@dataclass
class Config:
    da: DAConfig
    train: SourceTrainConfig
    model: ModelConfig
    dataset: OfficeHomeConfig


cs = ConfigStore.instance()
cs.store(group="train", name="source", node=SourceTrainConfig)
cs.store(group="da", name="da", node=DAConfig)
cs.store(group="dataset", name="office-home", node=OfficeHomeConfig)
cs.store(group="model", name="model", node=ModelConfig)
