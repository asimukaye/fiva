import os
from dataclasses import dataclass, field, asdict
import typing as t
from functools import partial
import yaml
from pathlib import Path
import torch
from logutils import auto_configure_device

SEED = 42
# SEED = 1
# SEED = 2
ROOT_PATH = "/home/asim.ukaye/fed_learning/datasets/"
# ROOT_PATH = "/home/asim.ukaye/fed_learning/datasets"
# ROOT_PATH = "/l/users/asim.ukaye/"

NNUNET_RAW = ROOT_PATH + "/nnUNet_raw"
NNUNET_PREPROCESSED = ROOT_PATH + "/nnUNet_preprocessed"
NNUNET_RESULTS = ROOT_PATH + "/nnUNet_results"
os.environ["nnUNet_raw"] = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
# os.environ["nnUNet_results"] = NNUNET_RESULTS
DEVICE = "auto"


DATASET_SHORTHANDS = {
    "amos": "Dataset001_amos",
    "kits": "Dataset002_KiTS2019",
    "ts": "Dataset003_TotalSegmentator",
    "lits": "Dataset004_LiTS",
    "btcv": "Dataset005_BTCV",
    "chaos": "Dataset007_chaos",
    "abd1k": "Dataset008_abdomenct1k",
    "l2r": "Dataset009_learn2reg",
    "liver": "Dataset010_Liver",
    "pancreas": "Dataset011_Pancreas",
    "spleen": "Dataset012_Spleen",
}

DEBUG_DATASETS_MAP = {
    "Dataset001_amos": "Dataset101_amos",
    "Dataset002_KiTS2019": "Dataset102_KiTS2019",
    "Dataset003_TotalSegmentator": "Dataset103_TotalSegmentator",
    "Dataset004_LiTS": "Dataset104_LiTS",
    "Dataset005_BTCV": "Dataset105_BTCV",
    "Dataset007_chaos": "Dataset107_chaos",
    "Dataset008_abdomenct1k": "Dataset108_abdomenct1k",
    "Dataset009_learn2reg": "Dataset109_learn2reg",
    "Dataset010_Liver": "Dataset110_Liver",
    "Dataset011_Pancreas": "Dataset111_Pancreas",
    "Dataset012_Spleen": "Dataset112_Spleen",
}

# Define the labels needed for each dataset
LABEL_SUBSETS = {
    "Dataset001_amos": [
        "spleen",
        "right kidney",
        "left kidney",
        "gall bladder",
        "esophagus",
        "liver",
        "stomach",
        # "aorta",
        "pancreas",
        "duodenum",
        "urinary_bladder",
    ],
    # "Dataset002_KiTS2019": ["right kidney", "left kidney"],
    "Dataset003_TotalSegmentator": [
        # "aorta", # Missing in TS
        "duodenum",
        "esophagus",
        "gall bladder",
        "left kidney",
        # "pancreas",
        "right kidney",
        "stomach",
        # "trachea",
        "urinary_bladder",
    ],  # 'inferior vena cava', 'liver', 'spleen',
    "Dataset004_LiTS": ["liver"],
    "Dataset005_BTCV": [
        "spleen",
        "right kidney",
        "left kidney",
        "liver",
        "stomach",
        # "aorta",
        # "inferior vena cava",
        "pancreas",
    ],
    # "Dataset007_chaos": ["liver"],
    "Dataset008_abdomenct1k": ["spleen", "liver", "pancreas"],
    "Dataset009_learn2reg": [
        "spleen",
        "right kidney",
        "left kidney",
        "gall bladder",
        "esophagus",
        "liver",
        "stomach",
        # "aorta",
        # "inferior vena cava",
        "pancreas",
    ],
    "Dataset010_Liver": ["liver"],
    "Dataset011_Pancreas": ["pancreas"],
    "Dataset012_Spleen": ["spleen"],
}

# Based on a batch size of 64
N_ITERATIONS_PER_FED_EPOCH_2D = {
    "Dataset001_amos": 200,
    # "Dataset002_KiTS2019": 50,
    "Dataset003_TotalSegmentator": 300,
    "Dataset004_LiTS": 50,
    "Dataset005_BTCV": 50,
    "Dataset007_chaos": 50,
    # "Dataset008_ctorg":,
    "Dataset008_abdomenct1k": 200,
    "Dataset009_learn2reg": 50,
    "Dataset010_Liver": 30,
    "Dataset011_Pancreas": 30,
    "Dataset012_Spleen": 30,
    "Dataset112_Spleen": 5,
}

TRUE_SAMPLES_PER_DATASET = {
    "Dataset011_Pancreas": 281,
    "Dataset012_Spleen": 41,
    "Dataset008_abdomenct1k": 1000,
    "Dataset005_BTCV": 30,
    "Dataset003_TotalSegmentator": 1139,
    "Dataset112_Spleen": 5,
    "Dataset101_amos": 5,
    "Dataset109_learn2reg": 5,
    "Dataset004_LiTS": 131,
    "Dataset009_learn2reg": 30,
    "Dataset001_amos": 300,
    "Dataset010_Liver": 131,
}

SAMPLES_PER_DATASET = {
    "Dataset001_amos": 300,
    # "Dataset002_KiTS2019": 210,
    "Dataset003_TotalSegmentator": 1139,
    "Dataset004_LiTS": 131,  # 131
    "Dataset005_BTCV": 30,  # 30 actually less but we increase weight because of number of labels
    # "Dataset006_bcv_cervix": 30,
    "Dataset007_chaos": 20,
    "Dataset008_abdomenct1k": 1000,
    "Dataset009_learn2reg": 30,
    "Dataset010_Liver": 130,  # 130
    "Dataset011_Pancreas": 280,  # 281
    "Dataset012_Spleen": 50,  # 41
}


SLICES_PER_DATASET = {
    "Dataset011_Pancreas": 26719,
    "Dataset012_Spleen": 3650,
    "Dataset008_abdomenct1k": 201528,
    "Dataset005_BTCV": 3779,
    "Dataset003_TotalSegmentator": 292285,
    "Dataset004_LiTS": 46933,
    "Dataset009_learn2reg": 7680,
    "Dataset001_amos": 41430,
    "Dataset010_Liver": 58638,
}


@dataclass
class TrainConfig:
    name: str = "nnUNetTrainer"
    epochs: int = 1
    lr: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 16
    device: str = "auto"
    dropout_p: float = 0.3

    def __post_init__(self):
        if self.device == "auto":
            self.device = auto_configure_device()
        self.device = torch.device(self.device)  # type: ignore


def default_dataset_list() -> list[str]:
    return list()


@dataclass
class DatasetConfig:
    name: str
    ids: t.List[str] = field(default_factory=default_dataset_list)
    subsample_fraction: float = 1.0


@dataclass
class nnUNetConfig:
    nnUNet_results: str = ROOT_PATH + "/nnUNet_results"
    nnUNet_n_proc_DA: int = 0  # Set to 0 for debugging
    config: str = "2d"
    unpack_dataset: bool = True

    def __post_init__(self):

        os.environ["nnUNet_n_proc_DA"] = str(self.nnUNet_n_proc_DA)


@dataclass
class Config:
    train: TrainConfig
    dataset: DatasetConfig
    nnunet: nnUNetConfig
    master_dataset: str
    model: str
    num_rounds: int
    num_iters: dict
    fold: int = 0
    name: str = "---"
    desc: str = ""
    label_subset: bool = True
    seed: int = SEED
    checkpoint_every: int = 50
    debug: bool = False
    cknum: str = "latest"

    def __post_init__(self):
        ## Define internal config variables here
        self.use_wandb = True


@dataclass
class FIVAConfig(Config):
    agg_mode: str = "scalar"
    grad_mode: str = "strong"


def set_debug_config(cfg: Config) -> Config:
    print("_____DEBUG MODE______\n")

    cfg.use_wandb = False
    cfg.debug = True

    cfg.dataset.subsample_fraction = 0.05
    # cfg.dataset.subsample_fraction = 1.0
    cfg.train.epochs = 1
    cfg.num_rounds = 2
    cfg.num_iters = {k: 2 for k in cfg.dataset.ids}
    # new_datasets = [DEBUG_DATASETS_MAP[ds] for ds in cfg.dataset.ids]
    # new_datasets = [DEBUG_DATASETS_MAP[cfg.dataset.ids[4]]]
    # cfg.dataset.ids = new_datasets
    return cfg


def get_fiva_config() -> Config:
    data_ids = [
        "Dataset003_TotalSegmentator",
        "Dataset004_LiTS",
        "Dataset005_BTCV",
        "Dataset008_abdomenct1k",
        "Dataset009_learn2reg",
        # "Dataset010_Liver",
        # "Dataset011_Pancreas",
        # "Dataset012_Spleen",
    ]

    return FIVAConfig(
        TrainConfig(batch_size=64, device=DEVICE),
        DatasetConfig(name="ensemble", ids=data_ids),
        nnUNetConfig(config="2d", nnUNet_n_proc_DA=12),
        master_dataset="Dataset003_TotalSegmentator",
        model="---",
        name="fiva",
        num_rounds=2001,
        num_iters={k: N_ITERATIONS_PER_FED_EPOCH_2D[k] for k in data_ids},
        # agg_mode = "scalar",
        agg_mode="full",
        # agg_mode = "layer",
        # grad_mode="strong",
        grad_mode = "parameter_variance",
        # grad_mode="unit_debug",
        # grad_mode="fedavg_fallback",
        # desc=f"FIVA config, full with 2k rounds, no msd datasets, using gradient variance rerun3 and full aggregation, seed 2",
        desc=f"FIVA config, full with 2k rounds, no msd datasets, using gradient varianc and full aggregation, seed 2, forgetting factor 0.7",
        # desc=f"FIVA config, unit debug with scalar agg with 2k rounds",
        # desc=f"FIVA config, fedavg fallback with 2k rounds",
    )


def get_default_config(strategy: str) -> Config:
    data_ids = [
        # "Dataset009_learn2reg",
        "Dataset003_TotalSegmentator",
        "Dataset004_LiTS",
        "Dataset005_BTCV",
        "Dataset008_abdomenct1k",
        "Dataset009_learn2reg",
        # "Dataset010_Liver",
        # "Dataset011_Pancreas",
        # "Dataset012_Spleen",
    ]
    return Config(
        TrainConfig(batch_size=64, device=DEVICE),
        DatasetConfig(name="ensemble", ids=data_ids),
        nnUNetConfig(config="2d", nnUNet_n_proc_DA=12),
        master_dataset="Dataset003_TotalSegmentator",
        model="---",
        name=strategy,
        num_rounds=2001,
        num_iters={k: N_ITERATIONS_PER_FED_EPOCH_2D[k] for k in data_ids},
        # num_rounds=3,
        # desc=f"Default {strategy} config, fiva with stronger variances, run with full variance, using state dictionaries instead of params ",
        # desc=f"{strategy} config, central training with 250 iterations, btcv template ",
        # desc=f"{strategy} config, fedavg modified to client updates like fiva, 2k rounds",
        desc=f"{strategy} config, no MSD datasets",
    )


def get_single_config(data_name) -> Config:
    data_ids = [DATASET_SHORTHANDS[data_name]]
    return Config(
        TrainConfig(batch_size=64, device=DEVICE),
        DatasetConfig(name=data_name, ids=data_ids),
        nnUNetConfig(config="2d", nnUNet_n_proc_DA=12),
        master_dataset=DATASET_SHORTHANDS[data_name],
        num_iters={k: N_ITERATIONS_PER_FED_EPOCH_2D[k] for k in data_ids},
        model="nnunet",
        name="single",
        num_rounds=501,  # Specifies the number of epochs in this case
        # num_rounds=3,
        desc=f"{data_ids[0]} Standalone training",
    )


def compile_config(mode: str, dataset: str = "ts") -> Config:
    """Compile the configuration dictionary"""
    print("Mode: ", mode)
    print("\n")

    match mode:
        case "federated":
            cfg = get_default_config("federated")
        case "fiva":
            # cfg = get_default_config("fiva")
            cfg = get_fiva_config()
        case "single":
            cfg = get_single_config(dataset)
        case "central":
            cfg = get_default_config("central")
        case _:
            raise NotImplementedError
    return cfg


def load_config(path: str) -> Config:
    cfg_file = Path(path) / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_file}")
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
    # print("Loaded config: ", cfg_dict)
    train_cfg = cfg_dict["train"]
    dataset_cfg = cfg_dict["dataset"]
    nnunet_cfg = cfg_dict["nnunet"]
    nnunet_cfg.pop("nnUNet_n_proc_DA")
    train_cfg.pop("device")
    for key in ["train", "dataset", "nnunet"]:
        cfg_dict.pop(key, None)

    if cfg_dict["name"] == "fiva":
        cfg = FIVAConfig(
            TrainConfig(**train_cfg),
            DatasetConfig(**dataset_cfg),
            nnUNetConfig(**nnunet_cfg, nnUNet_n_proc_DA=12),
            **cfg_dict,
        )
    else:
        cfg = Config(
            train=TrainConfig(**train_cfg),
            dataset=DatasetConfig(**dataset_cfg),
            nnunet=nnUNetConfig(**nnunet_cfg, nnUNet_n_proc_DA=12),
            **cfg_dict,
        )
    # print(cfg)

    return cfg
