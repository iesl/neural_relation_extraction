# input: model_type, dim
# output: a list of "wandb sweep" command


import os
import sys
import uuid
import copy
import wandb
import time
import json
import argparse
import subprocess
from pathlib import Path


def config_generation(model_type, path, task, data_format):

    sweep_config = {
        # "controller": {"type": "local"},
        "program": "src/main.py",
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "${args}",
            "--train",
            "--data_path=" + path,
            task,
            data_format,
            "--encoder_type=roberta-large",
            "--score_func=" + model_type,
            "--train_batch_size=2",
            "--test_batch_size=16",
            "--patience=5",
            "--log_interval=0.25",
            "--epochs=50",
            "--wandb",
        ],
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "Micro F1 dev"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform", "min": -16.0, "max": -7.0}, # 1e-7 ~ 1e-3
            "warmup": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "dim": {"values": [256, 512, 768, 1024]},
            "max_text_length": {"values": [32, 64, 128, 256]},
            "weight_decay":{"distribution": "log_uniform", "min": -18.4, "max": -2.3}, # 1e-8 ~ 1e-1
        },
    }

    if model_type == "box":
        sweep_config["parameters"]["intersection_temp"] = {
            "distribution": "log_uniform",
            "min": -9.2,
            "max": -0.69,
        }
        sweep_config["parameters"]["volume_temp"] = {
            "distribution": "log_uniform",
            "min": -2.3,
            "max": 2.3,
        }

    return sweep_config



def main(config):
    if config.multi_label:
        task = "--multi_label"
    else: 
        task = "--multi_class"
    if config.full_annotation:
        data_format = "--full_annotation"
    else:
        data_format = "--partial_annotation"
    sweep_config = config_generation(model_type=config.model_type, path=config.data_path, task=task, data_format=data_format)
    sweep_id = wandb.sweep(sweep_config, project="re")
    os.system(f"sh bin/launch_train_sweep.sh dongxu/re/{sweep_id} {config.partition} {config.max_run} {config.num_machine-1} {config.memory_per_run}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Project: Relation extraction")
    parser.add_argument("--partition", type=str, default="1080ti-long")
    parser.add_argument("--username", type=str, default="dongxuzhang")
    parser.add_argument("--max_run", type=int, default=50)
    parser.add_argument("--num_machine", type=int, default=20)
    parser.add_argument("--memory_per_run", type=int, default=30000)

    parser.add_argument("--data_path", type=str, default="data/tacred/")
    parser.add_argument("--model_type", type=str, default="box")
    parser.add_argument("--multi_label", action='store_true', default=False)
    parser.add_argument("--full_annotation", action='store_true', default=False)
    config = parser.parse_args()
    if config.model_type not in [
        "box",
        "nn",
        "biaffine"
    ]:
        raise Exception(f"model type {config.model_type} does not exist")
    main(config)