"""
R2R navigation agent for QwenGR00T_with_Language 14-dim models.

14-dim action layout: R2R occupies dims 3-6 (forward, left, right, stop).
The agent calls predict_action every step, takes the first action from the chunk,
slices dims 3:7, and does argmax to get the discrete habitat action.
"""

import os
import random

import cv2
import numpy as np
import torch
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from PIL import Image

from OneVLA.model.framework.base_framework import baseframework

# R2R action dims in the 14-dim unified space
R2R_ACTION_START = 3
R2R_ACTION_END = 7  # exclusive, so dims 3,4,5,6

# Onehot index -> habitat action ID
# index 0 (forward) -> 1, index 1 (left) -> 2, index 2 (right) -> 3, index 3 (stop) -> 0
ACTION_IDX_TO_ENV = {0: 1, 1: 2, 2: 3, 3: 0}
IDX_TO_NAME = {0: "forward", 1: "left", 2: "right", 3: "stop"}
STOP_IDX = 3  # index of stop in the 4-dim R2R action [forward, left, right, stop]


class QwenOFT_R2R_Agent(Agent):
    """
    R2R navigation agent for QwenGR00T_with_Language 14-dim models.
    Calls model every step (no action chunk caching).
    Slices R2R dims 3-6 from the 14-dim output.

    Stop threshold control via env vars (set to 0 to disable, fallback to argmax):
        ONEVLA_STOP_THRESHOLD   - stop dim value above which we consider "stop" (default: 0, disabled)
        ONEVLA_CHUNK_STOP_RATIO - fraction of chunk timesteps exceeding threshold to trigger stop (default: 0.3)
    """

    def __init__(
        self,
        checkpoint_path,
        result_path,
        require_map=True,
        device=None,
        max_history_frames=8,
    ):
        # Stop threshold config from env (0 = disabled, use argmax)
        self.stop_threshold = float(os.environ.get("ONEVLA_STOP_THRESHOLD", "0"))
        self.chunk_stop_ratio = float(os.environ.get("ONEVLA_CHUNK_STOP_RATIO", "0.3"))
        print(f"Initialize QwenGR00T_with_Language R2R Agent (14-dim)")
        print(f"  stop_threshold={self.stop_threshold}, chunk_stop_ratio={self.chunk_stop_ratio}")

        self.result_path = result_path
        self.require_map = require_map
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_frames = max_history_frames

        # Load model
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model = baseframework.from_pretrained(checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print("Initialization Complete")
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.step = 0
        self.episode_id = None
        self.reset()

    def _prepare_images(self):
        """Uniformly sample history frames + current frame for model input."""
        images = self.rgb_list.copy()
        if len(images) > self.max_history_frames:
            indices = np.linspace(0, len(images) - 1, self.max_history_frames, dtype=int).tolist()
            images = [images[i] for i in indices]

        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                pil_images.append(Image.fromarray(img).resize((224, 224)))
            else:
                pil_images.append(img.resize((224, 224)))
        return pil_images

    def _predict_action(self, instruction):
        """Run model inference, return full action chunk [chunk_size, action_dim]."""
        images = self._prepare_images()
        if len(images) == 0:
            return None

        with torch.no_grad():
            result = self.model.predict_action(
                batch_images=[images],
                instructions=[instruction],
                state=None,
            )

        normalized_actions = result.get("normalized_actions")
        if normalized_actions is None or normalized_actions.shape[1] == 0:
            return None

        return normalized_actions[0]  # [chunk_size, action_dim]

    def _onehot_to_action(self, action_vector):
        """Convert 4-dim onehot [forward, left, right, stop] to habitat action ID."""
        if action_vector is None or len(action_vector) < 4:
            return None

        action_vector = np.asarray(action_vector, dtype=np.float32)
        dominant_idx = int(np.argmax(action_vector))
        dominant_value = float(action_vector[dominant_idx])

        if dominant_value < 0.1:
            return None

        return ACTION_IDX_TO_ENV.get(dominant_idx, 1)

    def _addtext(self, image, instruction, navigation):
        """Add text overlay to visualization image."""
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        y_line = h + (50 + textsize[1]) // 2

        words = instruction.split(' ')
        x = 10
        line = ""
        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line
        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + textsize[1] + 10
        cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image

    def reset(self):
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id += 1
        self.step = 0

    def act(self, observations, info, episode_id):
        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        instruction_text = observations["instruction"]["text"]

        # Predict and extract R2R dims
        action_chunk = self._predict_action(instruction_text)

        if action_chunk is not None and action_chunk.shape[0] > 0:
            first_action = action_chunk[0]  # [action_dim]
            action_dim = len(first_action)

            # Debug: print full output to verify R2R dims
            if self.step < 5:
                print(f"[DEBUG] step={self.step} action_dim={action_dim}")
                print(f"[DEBUG] full_action={np.array2string(first_action, precision=4, suppress_small=True)}")
                print(f"[DEBUG] dims 3-6 (r2r): {first_action[3:7]}")

            # Helper to slice R2R dims from a single action vector
            def _slice_r2r(act):
                if len(act) >= 14:
                    return act[R2R_ACTION_START:R2R_ACTION_END]
                elif len(act) == 11:
                    return act[:4]
                else:
                    return act[:4]

            r2r_action = _slice_r2r(first_action)

            # Check stop: threshold on first step only (chunk voting disabled when ratio=0)
            chunk_stop = False
            if self.stop_threshold > 0:
                if self.chunk_stop_ratio > 0:
                    # Chunk voting mode: count how many steps exceed threshold
                    chunk_size = action_chunk.shape[0]
                    stop_count = 0
                    for t in range(chunk_size):
                        r2r_t = _slice_r2r(action_chunk[t])
                        if r2r_t[STOP_IDX] > self.stop_threshold:
                            stop_count += 1
                    stop_ratio = stop_count / chunk_size
                    if self.step < 5:
                        print(f"[DEBUG] chunk stop voting: {stop_count}/{chunk_size} = {stop_ratio:.2f} (threshold={self.stop_threshold}, ratio={self.chunk_stop_ratio})")
                    if stop_ratio >= self.chunk_stop_ratio:
                        chunk_stop = True
                else:
                    # Simple threshold mode: only check first step
                    if r2r_action[STOP_IDX] > self.stop_threshold:
                        chunk_stop = True
                    if self.step < 5:
                        print(f"[DEBUG] stop check: stop_val={r2r_action[STOP_IDX]:.4f} > {self.stop_threshold} = {chunk_stop}")

            if chunk_stop:
                action_to_run = ACTION_IDX_TO_ENV[3]  # stop
            else:
                action_id = self._onehot_to_action(r2r_action)
                if action_id is not None:
                    action_to_run = action_id
                else:
                    print(f"Warning: onehot conversion failed, r2r_action={r2r_action}, using random")
                    action_to_run = random.randint(1, 3)
                    r2r_action = None
        else:
            print("Warning: prediction failed, using random action")
            action_to_run = random.randint(1, 3)
            r2r_action = None

        if self.require_map:
            vector_text = "GR00T R2R[3:7]: {}".format(
                np.array2string(r2r_action, precision=2) if r2r_action is not None else "None"
            )
            img = self._addtext(output_im, instruction_text, vector_text)
            self.topdown_map_list.append(img)

        self.step += 1
        return {"action": action_to_run}
