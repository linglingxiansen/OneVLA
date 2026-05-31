#!/usr/bin/env python3
"""
R2R VLN-CE evaluation entry point for QwenOFT 28-dim models.
Simplified from NaVid-VLN-CE-main/run.py — only supports QwenOFT_CrossEmbodied.
"""

import argparse
import json
import os

import numpy as np
from habitat import Env
from habitat.datasets import make_dataset
from tqdm import trange
from VLN_CE.vlnce_baselines.config.default import get_config

from agent import QwenOFT_R2R_Agent


def main():
    parser = argparse.ArgumentParser(description="R2R eval for QwenOFT 28-dim models")
    parser.add_argument("--exp-config", type=str, required=True, help="Habitat experiment config yaml")
    parser.add_argument("--split-num", type=int, required=True, help="Number of parallel splits")
    parser.add_argument("--split-id", type=int, required=True, help="This split's ID")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint .pt path")
    parser.add_argument("--result-path", type=str, required=True, help="Directory to save results")
    parser.add_argument("--exp-save", type=str, default="video-data", help="'video-data' to save maps")
    args = parser.parse_args()

    config = get_config(args.exp_config)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)

    dataset_split = dataset.get_splits(args.split_num)[args.split_id]
    evaluate(config, args.split_id, dataset_split, args.model_path, args.result_path, args.exp_save)


def evaluate(config, split_id, dataset, model_path, result_path, exp_save):
    env = Env(config.TASK_CONFIG, dataset)
    require_map = "video" in (exp_save or "")

    agent = QwenOFT_R2R_Agent(
        checkpoint_path=model_path,
        result_path=result_path,
        require_map=require_map,
    )

    num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    for _ in trange(num_episodes, desc=f"split-{split_id}"):
        obs = env.reset()
        agent.reset()
        iter_step = 0
        continuse_rotation_count = 0
        last_dtg = 999

        while not env.episode_over:
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)

        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id

        if "data" in exp_save:
            log_dir = os.path.join(result_path, "log")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"stats_{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()
