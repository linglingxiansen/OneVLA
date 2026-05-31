from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
from transforms3d.euler import euler2axangle
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path


from OneVLA.model.tools import read_mode_config
# from OneVLA.model.framework.base_framework import baseframework

# Import baseframework for local inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ONEVLA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "OneVLA"))
if os.path.isdir(ONEVLA_ROOT) and ONEVLA_ROOT not in sys.path:
    sys.path.append(ONEVLA_ROOT)

import sys
import torch
from PIL import Image
from OneVLA.model.framework.base_framework import baseframework
import OneVLA.model.framework.QwenGR00T_with_Language  # noqa: F401


class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble = True,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10093,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "oxe_bridge" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            # if action_ensemble_horizon is None:
            #     # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
            #     action_ensemble_horizon = 7
            if action_ensemble_horizon is None:
                # 动态获取模型的 action chunk size，而不是硬编码为7
                action_ensemble_horizon = self.action_chunk_size  # 使用实际的chunk size
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "oxe_rt1" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps


        self.cfg_scale = cfg_scale # 1.5

        self.image_size = image_size
        self.action_scale = action_scale # 1.0
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))
        # image: Image.Image = Image.fromarray(image)

        image = self._resize_image(image)
        vla_input = {
            "batch_images": [[image]],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "cfg_scale": self.cfg_scale,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        response = self.client.infer(vla_input)
        
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        normalized_actions = normalized_actions[0]
        
        
        if normalized_actions.shape[-1] == 11:
            normalized_actions = normalized_actions[:, -7:]


         # 只使用前5个动作（如果chunk_size=8，则从8个动作中取前5个）
        if normalized_actions.shape[0] > 5:
            normalized_actions = normalized_actions[:5]

        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        # unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key) # 其实也是很环境 specific 的
        return norm_stats[unnorm_key]["action"]



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


class M1InferenceLocal:
    """
    Local inference class that directly loads the model using baseframework.from_pretrained,
    similar to libero's M1InferenceLocal, but adapted for SimplerEnv interface.
    """
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        device: Optional[str] = None,
        generate_text: bool = False,
    ) -> None:
        
        # Load model locally using baseframework.from_pretrained
        if policy_ckpt_path and os.path.exists(policy_ckpt_path):
            print(f"Loading GR00T checkpoint from: {policy_ckpt_path}")
            self.model = baseframework.from_pretrained(policy_ckpt_path)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Checkpoint loaded successfully!")
        elif policy_ckpt_path:
            raise FileNotFoundError(f"Checkpoint not found at {policy_ckpt_path}")
        else:
            raise ValueError("No checkpoint provided. Please provide a valid checkpoint path.")
        

        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)


        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "oxe_bridge" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            # if action_ensemble_horizon is None:
            #     action_ensemble_horizon = 7
            if action_ensemble_horizon is None:
                # 动态获取模型的 action chunk size，而不是硬编码为7
                action_ensemble_horizon = self.action_chunk_size  # 使用实际的chunk size
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "oxe_rt1" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported. The other datasets can be found in the huggingface config.json file."
            )
        
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.generate_text = generate_text

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None
        self.step_count = 0
        
        # Determine which dataset config to use based on unnorm_key
        self.dataset_key = "vla_data2" if unnorm_key == "franka" else "vla_data"
        
        # Get image size from the model config for the selected dataset
        model_config, _ = read_mode_config(policy_ckpt_path)
        dataset_cfg = model_config.get('datasets', {}).get(self.dataset_key, {})
        # Try to get image_size or default_image_resolution
        config_image_size = dataset_cfg.get('image_size', None)
        if config_image_size is None:
            default_image_resolution = dataset_cfg.get('default_image_resolution', None)
            if default_image_resolution is not None and isinstance(default_image_resolution, list) and len(default_image_resolution) >= 2:
                config_image_size = default_image_resolution[-2:]  # Take last 2 elements (H, W)
        
        # Use config image size if available, otherwise use provided image_size
        if config_image_size is not None:
            if isinstance(config_image_size, (list, tuple)):
                self.image_size = list(config_image_size)
            else:
                # If it's a single value, use it for both dimensions
                self.image_size = [config_image_size, config_image_size] if isinstance(config_image_size, (int, float)) else self.image_size
            print(f"*** Using image size from {self.dataset_key} config: {self.image_size} ***")
        else:
            print(f"*** Using default image size: {self.image_size} (no config found for {self.dataset_key}) ***")

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None
        self.step_count = 0

    def step(
        self, 
        image: np.ndarray, 
        task_description: Optional[str] = None, 
        step: int = 0,
        *args, 
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference using local model
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
            step: int, current step number (for action chunk caching)
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))
        
        # Resize image and convert to PIL Image
        image_resized = self._resize_image(image)
        if image_resized.dtype != np.uint8:
            image_resized = (image_resized * 255).astype(np.uint8) if image_resized.max() <= 1.0 else image_resized.astype(np.uint8)
        pil_image = Image.fromarray(image_resized)

        # Use internal step counter (simplerenv doesn't pass step parameter)
        # step parameter is only used if explicitly provided and > 0
        if step == 0:
            # Use internal counter
            step = self.step_count
        else:
            # If step is explicitly provided, use it and update internal counter
            self.step_count = step
        
        # ========== OLD CODE: 16步推理一次，使用动作块缓存 ==========
        # action_chunk_size = self.action_chunk_size
        # if step % action_chunk_size == 0:
        #     # Predict action chunk using local model
        #     with torch.no_grad():
        #         result = self.model.predict_action(
        #             batch_images=[[pil_image]],
        #             instructions=[self.task_description],
        #             state=None,
        #             generate_text=self.generate_text,
        #             dataset_key=self.dataset_key,
        #         )
        #     
        #     normalized_actions = result.get("normalized_actions")
        #     if normalized_actions is None or normalized_actions.shape[1] == 0:
        #         raise ValueError("Failed to get normalized actions from model")
        #     
        #     # normalized_actions shape: [B, chunk_size, action_dim]
        #     normalized_actions = normalized_actions[0]  # [chunk_size, action_dim]
        #     
        #     # Handle 11-dim actions (take last 7 dims)
        #     if normalized_actions.shape[-1] == 11:
        #         normalized_actions = normalized_actions[:, -7:]
        # 
        #     self.raw_actions = self.unnormalize_actions(
        #         normalized_actions=normalized_actions, 
        #         action_norm_stats=self.action_norm_stats
        #     )
        # 
        # raw_actions = self.raw_actions[step % action_chunk_size]  # 移除 [None]
        # 
        # if self.action_ensemble:
        #     raw_actions = self.action_ensembler.ensemble_action(raw_actions)
        # 
        # raw_actions = raw_actions[None]
        # ========== END OLD CODE ==========
        
        # ========== NEW CODE: 每步都预测一次，使用整个chunk进行ensemble ==========
        # Predict action chunk using local model (every step)
        with torch.no_grad():
            result = self.model.predict_action(
                batch_images=[[pil_image]],
                instructions=[self.task_description],
                state=None,
                generate_text=self.generate_text,
                dataset_key=self.dataset_key,
            )
        
        normalized_actions = result.get("normalized_actions")
        if normalized_actions is None or normalized_actions.shape[1] == 0:
            raise ValueError("Failed to get normalized actions from model")
        
        # normalized_actions shape: [B, chunk_size, action_dim]
        normalized_actions = normalized_actions[0]  # [chunk_size, action_dim]
        
        # Handle 11-dim actions (take last 7 dims)
        if normalized_actions.shape[-1] == 11:
            normalized_actions = normalized_actions[:, -7:]
        
        # 反归一化整个action chunk（与M1Inference保持一致）
        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, 
            action_norm_stats=self.action_norm_stats
        )
        
        # 使用整个chunk进行ensemble（与M1Inference保持一致）
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        else:
            # 如果不使用ensemble，取第一个动作
            raw_actions = raw_actions[0:1]
        # ========== END NEW CODE ==========


        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        self.step_count += 1
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        # Use q01/q99 for simplerenv (oxe_bridge, oxe_rt1), min/max for libero (franka)
        if "q01" in action_norm_stats and "q99" in action_norm_stats:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Get action normalization statistics from checkpoint.
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)

        unnorm_key = M1InferenceLocal._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config['framework']['action_model']['future_action_window_size'] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Check and validate unnorm_key.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)