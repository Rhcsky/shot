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
    batch_size: int = 128
    lr: float = 1e-2
    seed: int = 2020
    smooth: float = 0.1
    use_pretrained_backbone: bool = True


@dataclass
class TargetTrainConfig:
    worker: int = 4
    max_epoch: int = 15
    batch_size: int = 128
    lr: float = 1e-2
    seed: int = 2020
    smooth: float = 0.1
    saved_model_path: str = '../office-home/train_source_office-home_pda_08-16_19-10'

    gent: bool = True
    ent: bool = True
    threshold: int = 10
    cls_par: float = 0.3
    ent_par: float = 1.0
    lr_decay1: float = 0.1
    lr_decay2: float = 1.0


@dataclass
class ModelConfig:
    backbone: str = 'resnet50'
    layer: str = 'wn'
    classifier: str = 'bn'
    distance: str = 'cosine'
    bottleneck_dim: int = 256
    epsilon: float = 1e-5


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
    t: int = 1
    name_src: str = MISSING
    name_tar: str = MISSING


@dataclass
class RMFDatasetConfig(DAConfig):
    name: str = "RMFD"
    root: str = "../../data"
    resize_size: int = 112
    crop_size: int = 112

    if DAConfig.type == "oda":
        num_class: int = 442
        src_class: int = MISSING
        tar_class: int = MISSING
    else:
        num_class: int = 442
        src_class: int = 442
        tar_class: int = 442

    s: int = 0
    t: int = 1
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
cs.store(group="train", name="target", node=TargetTrainConfig)
cs.store(group="da", name="da", node=DAConfig)
cs.store(group="dataset", name="office-home", node=OfficeHomeConfig)
cs.store(group="dataset", name="RMFD", node=RMFDatasetConfig)
cs.store(group="model", name="model", node=ModelConfig)
