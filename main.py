import os
from argparse import ArgumentParser
import random
from pathlib import Path
from dataclasses import asdict
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from icecream import install, ic

install()
ic.configureOutput(includeContext=True)

from logutils import setup_output_dirs
from config import compile_config, set_debug_config, load_config, ROOT_PATH

# torch._dynamo.config.force_parameter_static_shapes = False


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"[SEED] Simulator global seed is set to: {seed}!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", "-m", default="federated")
    parser.add_argument("--dataset", "-i", default="ts")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--from_ckpt", type=str, default=None)
    parser.add_argument("--cknum", type=str, default="latest")

    args = parser.parse_args()

    if args.from_ckpt:
        cfg = load_config(args.from_ckpt)
        from_ckpt = True
        exp_path = Path(args.from_ckpt)
        cfg.cknum = args.cknum

        print(yaml.dump(asdict(cfg)))
        mode = cfg.name

    else:
        mode = args.mode

        from_ckpt = False

        cfg = compile_config(mode, args.dataset)
        if args.debug:
            cfg = set_debug_config(cfg)
            exp_path = setup_output_dirs(ROOT_PATH, "debug")
        else:
            exp_path = setup_output_dirs(ROOT_PATH, mode)
        cfg.nnunet.nnUNet_results = str(exp_path.resolve())

        print(yaml.dump(asdict(cfg)))

        with open("config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f)

    os.environ["nnUNet_results"] = str(exp_path)

    set_seed(cfg.seed)

    if mode == "federated":
        from train import train_fedavg

        train_fedavg(
            cfg,
            exp_path=exp_path,
            from_ckpt=from_ckpt,
        )
    elif mode == "fiva":
        from fiva import train_fiva

        train_fiva(
            cfg,
            exp_path=exp_path,
            from_ckpt=from_ckpt,
        )
    elif mode == "central":
        from centralized import train_centralized

        train_centralized(
            cfg,
            exp_path=exp_path,
            from_ckpt=from_ckpt,
        )
    elif mode == "single":
        from single import train_single

        train_single(
            cfg,
            exp_path=exp_path,
            from_ckpt=from_ckpt,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
