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


def config_generation(batchsize, score_func, classifier, path, task, data_format, contrastive, perturb):

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
            "--score_func=" + score_func,
            "--classifier=" + classifier,
            "--train_batch_size=" + str(batchsize),
            "--test_batch_size=16",
            "--patience=5",
            "--log_interval=0.25",
            "--epochs=10",
            "--max_text_length=512",
            "--dropout_rate=0.1",
            "--max_grad_norm=1.0",
            "--wandb",
            "--save_embedding",
        ],
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "Micro F1 dev"},
        "parameters": {
            # 1e-7 ~ 1e-3
            "learning_rate": {"distribution": "log_uniform", "min": -16.0, "max": -7.0},
            "grad_accumulation_steps": {"values": [1, 4, 16, 64]},
            "dim": {"values": [256, 512, 768, 1024]},
            # 1e-8 ~ 1e-1
            "weight_decay": {"distribution": "log_uniform", "min": -18.4, "max": -2.3},
        },
    }

    if task == "--multi_class":

        if score_func != "cosine":
            if classifier == "softmax_margin":
                sweep_config["parameters"]["margin"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 10.0}

            elif classifier in ["pairwise_margin", "pairwise_margin_na"]:
                sweep_config["parameters"]["margin_pos"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 10.0}
                sweep_config["parameters"]["margin_neg"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 10.0}

        elif score_func == "cosine":
            sweep_config["parameters"]["rescale_factor"] = {
                "distribution": "log_uniform",
                "min": 0,
                "max": 6.9}

            if classifier == "softmax_margin":
                sweep_config["parameters"]["margin"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.5}

            if classifier in ["pairwise_margin", "pairwise_margin_na"]:
                sweep_config["parameters"]["margin_pos"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.5}
                sweep_config["parameters"]["margin_neg"] = {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.5}

    if contrastive == True:

        sweep_config["parameters"]["contrastive_weight"] = {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0}

    if perturb == True:

        sweep_config["parameters"]["perturb_weight"] = {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0}

    if score_func == "box":
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
    sweep_config = config_generation(batchsize=config.batchsize, score_func=config.score_func, classifier=config.classifier,
                                     path=config.data_path, task=task, data_format=data_format, contrastive=config.contrastive, perturb=config.perturb)
    sweep_id = wandb.sweep(sweep_config, project="re")
    os.system(f"sh bin/launch_train_sweep.sh dongxu/re/{sweep_id} {config.partition} {config.max_run} {config.num_machine-1} {config.memory_per_run}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Project: Relation extraction")
    parser.add_argument("--partition", type=str, default="1080ti-long")
    parser.add_argument("--username", type=str, default="dongxuzhang")
    parser.add_argument("--max_run", type=int, default=50)
    parser.add_argument("--num_machine", type=int, default=20)
    parser.add_argument("--memory_per_run", type=int, default=30000)

    parser.add_argument("--data_path", type=str, default="data/tacred/")
    parser.add_argument("--multi_label", action='store_true', default=False)
    parser.add_argument("--full_annotation",
                        action='store_true', default=False)
    parser.add_argument("--score_func", type=str, default="dot")
    parser.add_argument("--classifier", type=str, default="softmax")
    parser.add_argument("--contrastive", action='store_true', default=False)
    parser.add_argument("--perturb", action='store_true', default=False)
    parser.add_argument("--batchsize", type=int, default=2)
    config = parser.parse_args()
    if config.score_func not in [
        "box",
        "dot",
        "cosine",
        "biaffine"
    ]:
        raise Exception(f"score_func {config.score_func} does not exist")

    if config.classifier not in [
        "softmax",
        "softmax_margin",
        "pairwise_margin",
        "pairwise_margin_na"
    ]:
        raise Exception(f"classifier {config.classifier} does not exist")

    main(config)
