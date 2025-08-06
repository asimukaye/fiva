import os
import time

import logging
import os

import torch
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path


def get_wandb_run_id(root_dir=".") -> str:
    list_dir = os.listdir(root_dir + "/wandb/latest-run")
    run_id = ""
    for list_item in list_dir:
        if ".wandb" in list_item:
            run_id = list_item.split("-")[-1].removesuffix(".wandb")
            print(f"Found run_id: {run_id}")
    # with open(root_dir + "/wandb/wandb-resume.json", "r") as f:
    #     wandb_json = json.load(f)
    # run_id = wandb_json["run_id"]
    return run_id


def setup_output_dirs(root_path , suffix="debug"):
    # os.makedirs("output", exist_ok=True)
    experiment_date = time.strftime("%y-%m-%d")
    experiment_time = time.strftime("%H-%M-%S")
    # output_dir = f"output/{experiment_date}/{suffix}/{experiment_time}"

    output_dir = Path(f"{root_path}/output") / experiment_date / suffix / experiment_time

    # output_dir = output_dir.resolve()

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logging to {output_dir}")

    # os.makedirs(output_dir, exist_ok=False)

    os.chdir(output_dir)
    return output_dir


def get_free_gpus(min_memory_reqd=4096):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    # min_memory_reqd = 10000
    ids = gpu_df.index[gpu_df["memory.free"] > min_memory_reqd]
    for id in ids:
        logging.debug(
            "Returning GPU:{} with {} free MiB".format(
                id, gpu_df.iloc[id]["memory.free"]
            )
        )
    return ids.to_list()


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    logging.debug("Returning GPU:{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"]))  # type: ignore
    return idx


def auto_configure_device():

    if torch.cuda.is_available():
        # Set visible GPUs
        # TODO: MAke the gpu configurable
        gpu_ids = get_free_gpus()
        # logging.info('Selected GPUs:')
        logging.info("Selected GPUs:" + ",".join(map(str, gpu_ids)))

        # Disabling below line due to cluster policies
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        if torch.cuda.device_count() > 1:
            device = f"cuda:{get_free_gpu()}"
        else:
            device = "cuda"

    else:
        device = "cpu"
    logging.info(f"Auto Configured device to: {device}")
    return device
    # return torch.device(device)
