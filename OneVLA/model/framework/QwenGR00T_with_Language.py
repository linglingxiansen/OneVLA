# Copyright 2025 OneVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Qwen-GR00T Framework with Language Generation
Outputs both text and actions:
1. First generates text tokens using Qwen-VL's language decoder
2. Then generates continuous actions using Flow-matching head
Format: <text>generated_text</text><action>continuous_actions</action>
"""
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from OneVLA.utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from OneVLA.model.framework.base_framework import baseframework
from OneVLA.model.modules.vlm import get_vlm_model
from OneVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from OneVLA.utils import resize_images
from OneVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenGR00T_with_Language")
class Qwen_GR00T_with_Language(baseframework):
    """
    Multimodal VLA model with language generation + flow-matching action head.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # Initialize VLM (Qwen-VL)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align dimensions for action model
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        # Initialize action model (Flow-matching head)
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # Action window configuration
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Loss weights
        self.language_loss_weight = config.framework.get("language_loss_weight", 1.0)
        self.action_loss_weight = config.framework.get("action_loss_weight", 1.0)

        # Special tokens for marking text and action sections
        self.text_start_token = config.framework.get("text_start_token", "<text>")
        self.text_end_token = config.framework.get("text_end_token", "</text>")
        self.action_start_token = config.framework.get("action_start_token", "<action>")
        self.action_end_token = config.framework.get("action_end_token", "</action>")

        self._add_special_tokens()

    def _add_special_tokens(self):
        special_tokens = [
            self.text_start_token,
            self.text_end_token,
            self.action_start_token,
            self.action_end_token,
        ]

        existing = set(self.qwen_vl_interface.processor.tokenizer.additional_special_tokens)
        new_tokens = [t for t in special_tokens if t not in existing]

        if new_tokens:
            num_added = self.qwen_vl_interface.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            if num_added > 0:
                self.qwen_vl_interface.model.resize_token_embeddings(
                    len(self.qwen_vl_interface.processor.tokenizer)
                )

        self.text_start_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(self.text_start_token)
        self.text_end_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(self.text_end_token)
        self.action_start_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(self.action_start_token)
        self.action_end_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(self.action_end_token)

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Training forward pass: language loss + action flow-matching loss.
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        text_responses = [example.get("text_response", "") for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        task_type = examples[0].get("task_type", None) if examples else None

        # Format: <text>response</text><action>
        formatted_responses = []
        for text_resp in text_responses:
            if text_resp:
                formatted_responses.append(f"{self.text_start_token}{text_resp}{self.text_end_token}{self.action_start_token}")
            else:
                formatted_responses.append(f"{self.action_start_token}")

        qwen_inputs_with_labels = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=formatted_responses
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs_with_labels,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            language_loss = self._compute_text_only_loss(
                qwenvl_outputs.logits,
                qwen_inputs_with_labels.get('labels', qwen_inputs_with_labels['input_ids']),
                qwen_inputs_with_labels['input_ids']
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        action_features = self._extract_action_features(
            last_hidden, qwen_inputs_with_labels['input_ids'], self.action_start_token_id
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            action_mask = None
            if "action_mask" in examples[0]:
                action_mask = torch.tensor(
                    np.array([ex["action_mask"] for ex in examples]), device=last_hidden.device, dtype=last_hidden.dtype
                )[:, -(self.future_action_window_size + 1):]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_features_repeated = action_features.repeat(repeated_diffusion_steps, 1, 1)
            action_mask_repeated = action_mask.repeat(repeated_diffusion_steps, 1) if action_mask is not None else None

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                action_features_repeated, actions_target_repeated, state_repeated,
                task_type=task_type, action_mask=action_mask_repeated
            )

        total_loss = self.language_loss_weight * language_loss + self.action_loss_weight * action_loss

        return {"language_loss": language_loss, "action_loss": action_loss, "total_loss": total_loss}

    def _extract_action_features(self, hidden_states, input_ids, action_token_id):
        """Extract features from hidden states (uses full sequence)."""
        batch_size = hidden_states.shape[0]
        action_features_list = []

        for i in range(batch_size):
            action_features_list.append(hidden_states[i, :, :])

        max_len = max(feat.shape[0] for feat in action_features_list)
        padded_features = []
        for feat in action_features_list:
            if feat.shape[0] < max_len:
                padding = torch.zeros(max_len - feat.shape[0], feat.shape[1], device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)

        return torch.stack(padded_features, dim=0)

    def _compute_text_only_loss(self, logits, labels, input_ids):
        """Compute cross-entropy loss only for text part (before <action> token)."""
        text_mask = torch.ones_like(labels, dtype=torch.bool)

        for i in range(labels.shape[0]):
            action_positions = (input_ids[i] == self.action_start_token_id).nonzero(as_tuple=True)[0]
            if len(action_positions) > 0:
                text_mask[i, action_positions[0].item():] = False

        masked_labels = labels.clone()
        masked_labels[~text_mask] = IGNORE_INDEX

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = masked_labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        generate_text: bool = True,
        max_text_length: int = 128,
        dataset_key: str = "vla_data",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Inference: optionally generate text, then predict actions via flow-matching.
        """
        dataset_cfg = getattr(self.config.datasets, dataset_key, None)
        if dataset_cfg is None:
            dataset_cfg = getattr(self.config.datasets, "vla_data", None)

        train_obs_image_size = None
        if dataset_cfg is not None:
            train_obs_image_size = getattr(dataset_cfg, "image_size", None)
            if train_obs_image_size is None:
                default_res = getattr(dataset_cfg, "default_image_resolution", None)
                if default_res is not None and isinstance(default_res, list) and len(default_res) >= 2:
                    train_obs_image_size = tuple(default_res[-2:])

        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        result = {}

        # Step 1: Generate text (optional)
        if generate_text:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = self.qwen_vl_interface.generate(**qwen_inputs, max_new_tokens=max_text_length, do_sample=False)
                input_length = qwen_inputs['input_ids'].shape[1]
                generated_text_ids = generated_ids[:, input_length:]
                result["generated_text"] = self.qwen_vl_interface.processor.batch_decode(generated_text_ids, skip_special_tokens=True)

        # Step 2: Get hidden states for action prediction
        action_prompts = [f"{inst} {self.action_start_token}" for inst in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=action_prompts)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        action_features = self._extract_action_features(last_hidden, qwen_inputs['input_ids'], self.action_start_token_id)

        # Step 3: Predict actions
        state_tensor = (
            torch.from_numpy(np.array(state)).to(action_features.device, dtype=action_features.dtype)
            if state is not None else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(action_features, state_tensor)

        result["normalized_actions"] = pred_actions.detach().cpu().numpy()
        return result
