# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import logging
import copy
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score.tool_memory import ToolMemory
from verl.utils.reward_score import tool_update_by_llm
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type
from datetime import datetime

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]



def convert_to_ndarray(data: Any, expected_batch_size: int) -> np.ndarray:
    """
    转换数据为np.ndarray，并强制检查shape[0]是否等于expected_batch_size
    """
    if isinstance(data, np.ndarray):
        if data.shape[0] != expected_batch_size:
            raise ValueError(
                f"数组 {data} 的shape[0] 应为 {expected_batch_size}，实际 {data.shape[0]}"
            )
        return data
    elif isinstance(data, list):
        # 列表长度必须等于批次大小（不足则填充，过长则截断）
        if len(data) < expected_batch_size:
            data += [data[-1]] * (expected_batch_size - len(data))  # 用最后一个元素填充
        elif len(data) > expected_batch_size:
            data = data[:expected_batch_size]  # 截断
        # 处理字符串列表（保留原始类型）
        if all(isinstance(item, str) for item in data):
            return np.array(data, dtype=object)
        else:
            return np.array(data)
    elif isinstance(data, str):
        # 单个字符串扩展为长度batch_size的数组（dtype=object）
        return np.array([data] * expected_batch_size, dtype=object)
    elif isinstance(data, (int, float)):
        # 标量扩展为长度batch_size的数组
        return np.array([data] * expected_batch_size)
    else:
        # 其他类型转字符串（根据实际需求调整）
        return np.array([str(data)] * expected_batch_size, dtype=object)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, test_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, test_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        if test_dataset is None:
            test_dataset = create_rl_dataset(self.config.data.test_files, self.config.data, self.tokenizer, self.processor)
        
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        #这里添加一个test_dataloader
        self.test_dataloader = StatefulDataLoader(
            dataset=self.test_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        assert len(self.test_dataloader) >= 1, "Test dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}, Size of test dataloader: {len(self.test_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _test(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.test_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _log_batch_state(self, batch, step_name, debug_path, max_samples=2):
        """动态记录batch的所有张量字段内容到日志文件，区分token字段和非token字段"""
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== [BATCH_STATE: {step_name}] =====\n")

            # 记录batch.batch（张量数据）的结构和全部内容
            f.write("--- batch.batch 结构 ---\n")
            tensor_keys = list(batch.batch.keys())
            f.write(f"包含的张量字段: {tensor_keys}\n")
            
            # 定义可解码的token字段白名单（仅这些字段尝试解码）
            decode_whitelist = {"input_ids", "responses", "raw_prompt_ids"}
            
            for key in tensor_keys:  # 遍历当前batch的所有张量字段
                value = batch.batch[key]
                f.write(f"\n字段 [{key}] 信息:\n")
                f.write(f"  形状: {value.shape if hasattr(value, 'shape') else '非张量'}\n")
                f.write(f"  数据类型: {value.dtype if hasattr(value, 'dtype') else '非张量'}\n")
                
                if key in decode_whitelist:
                    # 对token字段尝试解码（如input_ids, responses）
                    f.write(f"  [解码尝试] 开始尝试解码字段 [{key}]...\n")
                    try:
                        sample_ids = value[:max_samples].cpu().numpy()
                        f.write(f"  [解码调试] 提取样本形状: {sample_ids.shape}\n")
                        
                        valid_samples = []
                        for i, ids in enumerate(sample_ids):
                            # 跳过全为填充值的样本（假设0是填充值，可根据实际修改）
                            if len(ids[ids != 0]) == 0:
                                f.write(f"  [解码跳过] 样本{i}全为填充值，跳过解码\n")
                                continue
                            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                            valid_samples.append(f"样本{i}: {decoded[-1000:]}...")  # 截断长文本
                        
                        if valid_samples:
                            f.write(f"  前{len(valid_samples)}条有效解码文本: {valid_samples}\n")
                        else:
                            f.write("  [解码结果] 无有效样本，跳过解码\n")
                    except Exception as e:
                        f.write(f"  [解码错误] 解码失败，原因: {str(e)}\n")
                else:
                    # 对非token字段（如position_ids, attention_mask）记录数值
                    f.write(f"  [数值记录] 字段为非token类型，直接记录数值...\n")
                    try:
                        sample_values = value[:max_samples].cpu().numpy()
                        f.write(f"  [数值调试] 提取样本形状: {sample_values.shape}\n")
                        
                        numeric_samples = []
                        for i, vals in enumerate(sample_values):
                            # 仅记录前20个数值（避免长序列刷屏）
                            numeric_samples.append(f"样本{i}: {vals[:20].tolist()}...")
                        
                        f.write(f"  前{len(numeric_samples)}条数值样本: {numeric_samples}\n")
                    except Exception as e:
                        f.write(f"  [数值错误] 记录失败，原因: {str(e)}\n")

                # 记录非张量字段（保持原逻辑）
            f.write("\n\n--- batch.non_tensor_batch 结构 ---\n")
            non_tensor_keys = list(batch.non_tensor_batch.keys())
            f.write(f"包含的非张量字段: {non_tensor_keys}\n")
            
            for key in non_tensor_keys:
                value = batch.non_tensor_batch[key]
                f.write(f"\n字段 [{key}] 信息:\n")
                f.write(f"  类型: {type(value)}\n")
                if isinstance(value, (list, np.ndarray)):
                    samples = value[:max_samples] if len(value) > max_samples else value
                    f.write(f"  前{max_samples}条样本: {samples}\n")
                else:
                    f.write(f"  样本内容: {value}\n")

            # 记录元信息（保持原逻辑）
            f.write("\n\n--- 元信息 ---\n")
            batch_size = len(batch.batch.get("input_ids", []))  # 用input_ids推断batch_size
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"meta_info: {batch.meta_info}\n")
            f.write(f"===== [BATCH_STATE: {step_name}] 结束 =====\n\n")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        str1 = """ """
        memory = ToolMemory(str1)
        
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        SCORE_LOG_PATH = self.config.data.score_log_path
        DEBUG_LOG_PATH = self.config.data.debug_log_path
        try:
            # 构造日志条目
            step_entry = { 
                "val_global_steps": self.global_steps
                }

            # 写入JSON Lines文件（每行一个独立JSON对象）
            with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(step_entry, f, ensure_ascii=False)
                f.write("\n")  # 换行分隔不同条目
            print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(step_entry, f, ensure_ascii=False)
                f.write("\n")  # 换行分隔不同条目
            print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
        except Exception as e:
            print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            
        SCORE_LOG_PATH = self.config.data.score_log_path
        DEBUG_LOG_PATH = self.config.data.debug_log_path
        try:
            # 构造日志条目
            step_entry = { 
                "test_global_steps": self.global_steps
                }

            # 写入JSON Lines文件（每行一个独立JSON对象）
            with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(step_entry, f, ensure_ascii=False)
                f.write("\n")  # 换行分隔不同条目
            print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump(step_entry, f, ensure_ascii=False)
                f.write("\n")  # 换行分隔不同条目
            print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
        except Exception as e:
            print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")
        test_metrics = self._test()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        

        for epoch in range(self.config.trainer.total_epochs):
            
            

            for batch_dict in self.train_dataloader:
                SCORE_LOG_PATH = self.config.data.score_log_path
                DEBUG_LOG_PATH = self.config.data.debug_log_path
                try:
                    # 构造日志条目
                    step_entry = { 
                        "train_global_steps": self.global_steps
                        }

                    # 写入JSON Lines文件（每行一个独立JSON对象）
                    with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                        json.dump(step_entry, f, ensure_ascii=False)
                        f.write("\n")  # 换行分隔不同条目
                    print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
                    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                        json.dump(step_entry, f, ensure_ascii=False)
                        f.write("\n")  # 换行分隔不同条目
                    print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
                except Exception as e:
                    print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")

                # #下面的部分是步间更新####
                # print("\n====正在使用大模型更新工具描述====")
                # try:
                #     # 调用工具更新函数（传入文件路径和模型路径）
                #     tool_update_by_llm.process_data(
                #         file_path="",
                #         model_path=""
                #     )
                #     file_path=""
                #     print("\n====完成步间更新，正在清空实时数据文件====")
                #     # 工具更新完成后，清空实时数据文件
                #     if os.path.exists(file_path):
                #         with open(file_path, "w") as f:
                #             f.truncate(0)  # 清空文件内容
                #         print(f"已清空实时数据文件: {file_path}")
                #     else:
                #         print(f"数据文件不存在（{file_path}），无需清空")
                #     print("\n====已清空实时数据文件====")

                # except Exception as e:
                #     print(f"警告：工具更新或文件清空失败，原因: {str(e)}，主流程继续...")


                metrics = {}
                timing_raw = {}
                ##这个位置是看要不要生成描述的
                #batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_original: DataProto = DataProto.from_single_dict(batch_dict)
                batch = self._re_tokenize_with_dynamic_prompts(batch_original)
                #self._log_batch_state(batch_original, "INITIAL_BATCH", DEBUG_PATH)  # 新增

                # self._save_data_proto_prompts(
                #     data_proto=batch_original,
                #     prefix="original_prompt",
                #     save_dir="prompt_validation"
                # )
                # === 动态Prompt处理 ===
                # 1. 重新分词生成动态Prompt的input_ids（内部已调用_generate_dynamic_prompts）

                
                #self._log_batch_state(batch, "DYNAMIC_BATCH", DEBUG_PATH)  # 新增

                # self._save_data_proto_prompts(
                #     data_proto=batch,
                #     prefix="processed_prompt",
                #     save_dir="prompt_validation"
                # )
            
               
                # # 数据重复操作（此时dynamic_prompt长度应与batch一致）
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                #self._log_batch_state(batch, "AFTER_POP_GEN_BATCH", DEBUG_PATH)  # 新增

                # # 打印gen_batch中的input_ids解码结果（必须为"123"）
                # if "input_ids" in gen_batch.batch:
                #     decoded = [
                #         self.tokenizer.decode(ids, skip_special_tokens=True)
                #         for ids in gen_batch.batch["input_ids"][:3].cpu().numpy()
                #     ]
                #     print(f"[GEN_BATCH_DEBUG] gen_batch.input_ids解码: {decoded}")
                # else:
                #     print("[GEN_BATCH_DEBUG] gen_batch中无input_ids，模型将使用默认输入！")

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            #self._log_batch_state(batch=gen_batch, step_name="GENERATION_INPUT", debug_path=DEBUG_PATH)
                            #self._log_batch_state(batch=gen_batch_output, step_name="GENERATION_OUTPUT", debug_path=DEBUG_PATH)

                            
                            
                            
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    #self._log_batch_state(batch, "AFTER_ADD_UID", DEBUG_PATH)  # 新增
                    
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    #self._log_batch_state(batch, "AFTER_REPEAT", DEBUG_PATH)  # 新增
                    
                    batch = batch.union(gen_batch_output)
                    #self._log_batch_state(batch, "AFTER_UNION_GEN_OUTPUT", DEBUG_PATH)  # 新增
                    
                    

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    #self._log_batch_state(batch, "AFTER_ADD_RESPONSE_MASK", DEBUG_PATH)  # 新增
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    #self._log_batch_state(batch, "FINAL_BATCH_STATE", DEBUG_PATH) 

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            SCORE_LOG_PATH = self.config.data.score_log_path
                            DEBUG_LOG_PATH = self.config.data.debug_log_path
                            
                            try:
                                # 构造日志条目
                                step_entry = { 
                                    "validate_global_steps": self.global_steps
                                    }

                                # 写入JSON Lines文件（每行一个独立JSON对象）
                                with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                                    json.dump(step_entry, f, ensure_ascii=False)
                                    f.write("\n")  # 换行分隔不同条目
                                print(f"[DEBUG LOG] 已记录step数据到 {SCORE_LOG_PATH}")
                                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                                    json.dump(step_entry, f, ensure_ascii=False)
                                    f.write("\n")  # 换行分隔不同条目
                                print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
                            except Exception as e:
                                print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")

                            val_metrics: dict = self._validate()

                            SCORE_LOG_PATH = self.config.data.score_log_path
                            DEBUG_LOG_PATH = self.config.data.debug_log_path
                            try:
                                # 构造日志条目
                                step_entry = { 
                                    "test_global_steps": self.global_steps
                                    }

                                # 写入JSON Lines文件（每行一个独立JSON对象）
                                with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                                    json.dump(step_entry, f, ensure_ascii=False)
                                    f.write("\n")  # 换行分隔不同条目
                                print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
                                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                                    json.dump(step_entry, f, ensure_ascii=False)
                                    f.write("\n")  # 换行分隔不同条目
                                print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
                            except Exception as e:
                                print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")
                            test_metrics: dict = self._test()
                            print(f"test_metrics={test_metrics}")
                            
                            if is_last_step:
                                last_val_metrics = val_metrics
                                #这里加了一个最终的test集上的测试
                                # SCORE_LOG_PATH = self.config.data.score_log_path
                                # DEBUG_LOG_PATH = self.config.data.debug_log_path
                                # try:
                                #     # 构造日志条目
                                #     step_entry = { 
                                #         "test_global_steps": self.global_steps
                                #         }

                                #     # 写入JSON Lines文件（每行一个独立JSON对象）
                                #     with open(SCORE_LOG_PATH, "a", encoding="utf-8") as f:
                                #         json.dump(step_entry, f, ensure_ascii=False)
                                #         f.write("\n")  # 换行分隔不同条目
                                #     print(f"[DEBUG LOG] 已记录调试数据到 {SCORE_LOG_PATH}")
                                #     with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                                #         json.dump(step_entry, f, ensure_ascii=False)
                                #         f.write("\n")  # 换行分隔不同条目
                                #     print(f"[DEBUG LOG] 已记录step数据到 {DEBUG_LOG_PATH}")
                                # except Exception as e:
                                #     print(f"[DEBUG LOG ERROR] 日志记录失败: {str(e)}")
                                # test_metrics: dict = self._test()
                                # print(f"test_metrics={test_metrics}")
                                
                        metrics.update(val_metrics)


                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1


                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                

    # def _save_data_proto_prompts(self, data_proto: DataProto, prefix: str = "prompt", save_dir: str = "prompt_validation", skip_special_tokens: bool = True) -> str:
    #     tokenizer = self.tokenizer
    #     # 确保保存目录存在
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 生成带时间戳的文件名
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     file_path = os.path.join(save_dir, f"{prefix}_{timestamp}.txt")
        
    #     # 提取批次大小
    #     batch_size = len(data_proto)
        
    #     # 初始化存储内容
    #     save_content = [f"=== DataProto 验证 ===\n", 
    #                 f"批次大小: {batch_size}\n\n"]
        
    #     # 提取full_prompts字段
    #     full_prompts = []
    #     if "full_prompts" in data_proto.non_tensor_batch:
    #         # 处理np.ndarray类型
    #         if isinstance(data_proto.non_tensor_batch["full_prompts"], np.ndarray):
    #             full_prompts = data_proto.non_tensor_batch["full_prompts"].tolist()
    #         # 处理其他类型（如list）
    #         else:
    #             full_prompts = list(data_proto.non_tensor_batch["full_prompts"])
    #     else:
    #         full_prompts = ["[无full_prompts字段]" for _ in range(batch_size)]
        
    #     # 提取input_ids张量并解码
    #     input_ids_decoded = []
    #     if data_proto.batch is not None and "input_ids" in data_proto.batch:
    #         input_ids = data_proto.batch["input_ids"].cpu()
    #         for ids in input_ids:
    #             decoded = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    #             input_ids_decoded.append(decoded)
    #     else:
    #         input_ids_decoded = ["[无input_ids张量]" for _ in range(batch_size)]
        
    #     # 组装保存内容
    #     for i in range(batch_size):
    #         save_content.append(f"样本 {i}:\n")
    #         save_content.append(f"  full_prompts: {full_prompts[i]}\n")
    #         save_content.append(f"  input_ids解码: {input_ids_decoded[i]}\n")
    #         save_content.append("-" * 80 + "\n")
        
    #     # 写入文件
    #     with open(file_path, "w", encoding="utf-8") as f:
    #         f.writelines(save_content)
        
    #     print(f"已保存至: {file_path}")
    #     return file_path

    def _re_tokenize_with_dynamic_prompts(self, batch: DataProto) -> DataProto:
        """
        动态生成Prompt并处理数据，确保输出的DataProto符合原始类约束（TensorDict类型、批量对齐）
        """
        # ------------------------ 配置与初始化 ------------------------
        tokenizer = self.tokenizer
        config = self.config.data
        batch_size = len(batch)  

        # ------------------------ 1. 动态生成Prompt ------------------------
        dynamic_prompts = self._generate_dynamic_prompts(batch)  

        # ------------------------ 2. 分词与后处理（保持为torch.Tensor） ------------------------
        encoded = tokenizer(
            dynamic_prompts,
            max_length=config.max_prompt_length,
            truncation="longest_first",
            padding="max_length",
            return_tensors="pt",  # 返回PyTorch张量
            add_special_tokens=True
        )

        # 后处理（确保输出为torch.Tensor，不转numpy）
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=config.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=config.truncation
        )

        # 生成position_ids（保持为torch.Tensor）
        position_ids = compute_position_id_with_mask(attention_mask)
        # if not isinstance(position_ids, torch.Tensor):
        #     position_ids = torch.tensor(position_ids, dtype=torch.long)

        # ------------------------ 3. 构造TensorDict作为新的batch ------------------------
        # 直接使用torch.Tensor构造TensorDict（关键修复：batch必须是TensorDict类型）
        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        # 验证张量的批量大小（所有张量的dim0必须等于batch_size）
        for key, tensor in tensors.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"张量 {key} 的批量大小错误："
                    f"预期 {batch_size}（原始批次大小），实际 {tensor.shape[0]}"
                )

        # 构造TensorDict，指定batch_size（元组格式，与原始DataProto兼容）
        new_batch = TensorDict(
            source=tensors,
            batch_size=(batch_size,)  # 必须为元组（num_batch_dims=1）
        )

        # 遍历原始non_tensor_batch，逐个转换并检查批量大小
        new_non_tensor_batch: Dict[str, np.ndarray] = {}
        for key, value in batch.non_tensor_batch.items():
            new_non_tensor_batch[key] = convert_to_ndarray(
                data=value,
                expected_batch_size=batch_size
            )

        dynamic_raw_prompt_ids = []
        for prompt in dynamic_prompts:
            # 不添加特殊token（与RLHFDataset的raw_prompt_ids生成逻辑一致）
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            original_length = len(prompt_ids)

            # 截断逻辑（与RLHFDataset完全一致）
            if original_length > config.max_prompt_length:
                if config.truncation == "left":
                    prompt_ids = prompt_ids[-config.max_prompt_length:]
                elif config.truncation == "right":
                    prompt_ids = prompt_ids[:config.max_prompt_length]
                elif config.truncation == "middle":
                    left_half = config.max_prompt_length // 2
                    right_half = config.max_prompt_length - left_half
                    prompt_ids = prompt_ids[:left_half] + prompt_ids[-right_half:]
                elif config.truncation == "error":
                    raise RuntimeError(
                        f"动态提示长度{original_length}超过最大限制{config.max_prompt_length}（truncation=error）"
                    )

            dynamic_raw_prompt_ids.append(prompt_ids)

        # 转换为np.ndarray(dtype=object)（与collate_fn处理raw_prompt_ids的格式一致）
        new_non_tensor_batch["raw_prompt_ids"] = np.array(dynamic_raw_prompt_ids, dtype=object)
        if new_non_tensor_batch["raw_prompt_ids"].shape[0] != batch_size:
            raise ValueError(
                f"raw_prompt_ids批量大小错误：预期{batch_size}，实际{new_non_tensor_batch['raw_prompt_ids'].shape[0]}"
            )

        # 动态生成的full_prompts必须为np.ndarray（dtype=object，shape[0]=batch_size）
        new_non_tensor_batch["full_prompts"] = np.array(dynamic_prompts, dtype=object)
        if new_non_tensor_batch["full_prompts"].shape[0] != batch_size:
            raise ValueError(
                f"full_prompts 批量大小错误："
                f"预期 {batch_size}，实际 {new_non_tensor_batch['full_prompts'].shape[0]}"
            )

        # ------------------------ 5. 处理meta_info（直接继承原始值） ------------------------
        new_meta_info = copy.deepcopy(batch.meta_info)  # 深拷贝避免修改原数据
        
        
        return DataProto(
            batch=new_batch,  # 核心修复：batch是TensorDict而非普通字典
            non_tensor_batch=new_non_tensor_batch,
            meta_info=new_meta_info
        )

    def _generate_dynamic_prompts(self, batch: DataProto) -> list[str]:
        print("启动动态prompt生成，当前global_steps:", self.global_steps)
    
        TOOL_LOG_PATH = self.config.data.tool_log_path
        
       
        original_prompts = batch.non_tensor_batch.get("full_prompts", [""] * len(batch))
        
        
        dynamic_prompts = []
        memory = ToolMemory()
        fallback_count = 0
        current_tool_descs_str = memory.get_all_tools()
        #print(f"[TOOL_MEMORY_DEBUG] current_tool_descs_str内容:")
        #print(current_tool_descs_str)
        with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[TOOL_MEMORY_DEBUG][Step {self.global_steps}] current_tool_descs_str内容:\n")
            f.write(current_tool_descs_str + "\n" if current_tool_descs_str else "[空字符串]\n")
        tokenizer = self.tokenizer

        # 定义各模块标签（与原始Prompt结构顺序一致）
        tool_info_tag = "# 工具信息\n"
        car_info_tag = "\n\n# 进线车辆信息\n"
        requirement_tag = "\n\n# 要求\n"
        language_style_tag = "\n\n# 语言风格\n"
        output_format_tag = "\n\n# 输出格式\n"
        current_time_tag = "\n\n# 当前时间\n"

        # 新增的要求说明（插入到"# 要求"末尾，"# 语言风格"前）
        new_requirement = """
    - 建议你基于 # 工具信息 中已有的工具（不可编造新工具），生成或修改其描述（如调整语言表述、补充细节、增加示例），以帮助你更高效地理解和使用工具，将会有助于你学习工具的使用。修改后的描述需与“# 工具信息”中的格式一致（如以“## 工具名”开头，包含“工具名称”“工具描述”“需要填入”“使用示例”等字段），关键信息点不可随意修改,包括## 后面的工具名字段，不可更改，最好只修改你在本次对话中使用到的工具描述。在工具描述部分要确保按照格式输出内容，不要加无关解释。
        """.strip()

        # 新增的输出格式示例（插入到"# 输出格式"末尾，"# 当前时间"前）
        new_output_format = """
    - 默认需要调用工具，先输出思考过程，再参考工具信息中的使用示例输出完整的工具调用信息，输出示例：
    <thought>
    思考内容
    </thought>
    <actions>
    [search_used_cars({"query": "宝马三系"}), dcar_search({"query": "宝马三系和奔驰C级优势对比"})]
    </actions>
    <description>
    ## 查询二手车库存\n工具名称：search_used_cars\n工具描述：可以通过查询内容和筛选条件对于库存内的车辆进行筛选，并展示数条搜索结果\n需要填入：\nquery (str)：查询的内容，可以类似“奔驰”，“增程式”这类\nfilters (dict)：筛选项，具体可选的筛选项为：\n    price_range (List[int])：价格区间（单位是万元），用[15, 21]表示需要筛选15万到21万的车辆\n    mileage (List[int])：里程区间（单位是万公里），用[1, 6]表示需要筛选1万公里到6万公里里程的车辆\n    car_age (List[int])：车龄（单位是年），用[2, 4]表示需要筛选车龄为两年以上四年以下的车辆\n    energy_type (str)：能源类型，必须是[\"新能源\", \"非新能源\"]之一，纯电/插混/增程均属于\"新能源\"\n    category (List[str])：车辆级别，必须是[\"轿车\", \"SUV\", \"MPV\", \"跑车\"]中的一个或多个\n    emission_standard (List[str])：排放标准，必须是[\"国四\", \"国五\", \"国六\", \"国六b\"]中的一个或多个\npage (int)：当前页码，默认值为1\npage_size (int)：每页检索条数，默认值为5，最大值为30\n使用示例：\n    search_used_cars({\"query\": \"mini\", \"filters\": {\"car_age\": [1, 4]}})\n    search_used_cars({\"query\": \"宝马三系\", \"page_size\": 20})\n    search_used_cars({\"filters\": {\"energy_type\": \"新能源\", \"category\": [\"轿车\", \"跑车\"]}, \"page_size\": 30})\n\n
    </description>
    
        """.strip()

        for prompt in original_prompts:
            # 步骤1：替换工具信息段落（工具信息→进线车辆信息）
            tool_start = prompt.find(tool_info_tag)
            if tool_start != -1:
                tool_end = prompt.find(car_info_tag, tool_start)
                if tool_end == -1:
                    # 工具信息无结束标签（边界情况）
                    tool_section = f"{tool_info_tag}{current_tool_descs_str}"
                    dynamic_prompt = prompt[:tool_start] + tool_section
                else:
                    # 正常替换工具信息段落
                    tool_section = f"{tool_info_tag}{current_tool_descs_str}"
                    dynamic_prompt = prompt[:tool_start] + tool_section + prompt[tool_end:]
            else:
                dynamic_prompt = prompt  # 无工具信息段落，保留原始内容

            # 步骤2：在"# 要求"末尾插入新增要求（要求→语言风格）
            req_start = dynamic_prompt.find(requirement_tag)
            if req_start != -1:
                # 定位"# 要求"的结束位置（即下一个模块"# 语言风格"的起始位置）
                req_end = dynamic_prompt.find(language_style_tag, req_start)
                if req_end == -1:
                    # 无"# 语言风格"标签（边界情况），要求段落结束于Prompt末尾
                    req_end = len(dynamic_prompt)
                # 在"# 要求"现有内容后插入新要求（保持列表格式）
                dynamic_prompt = (
                    dynamic_prompt[:req_end] + 
                    "\n- " + new_requirement.replace("\n", "\n- ") +  # 与原有要求列表格式一致
                    dynamic_prompt[req_end:]
                )

            # 步骤3：在"# 输出格式"末尾插入新增示例（输出格式→当前时间）
            fmt_start = dynamic_prompt.find(output_format_tag)
            if fmt_start != -1:
                # 定位"# 输出格式"的结束位置（即下一个模块"# 当前时间"的起始位置）
                current_time_start = dynamic_prompt.find(current_time_tag)
                # 用新内容替换原内容
                dynamic_prompt = (
                    dynamic_prompt[:fmt_start] +
                    output_format_tag +
                    new_output_format +
                    dynamic_prompt[current_time_start:]
                )



            # 新增：动态Prompt长度检查与回退逻辑
            # 计算动态Prompt的token长度
            config = self.config.data
            tokenized_length = len(tokenizer.tokenize(dynamic_prompt))
            original_tokenized_length = len(tokenizer.tokenize(prompt))
            max_prompt_length = config.max_prompt_length
            if tokenized_length > max_prompt_length:
                dynamic_prompt = prompt  # 回退到原始Prompt
                fallback_count += 1
                # with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                #     f.write(f"[FALLBACK][Step {self.global_steps}] 样本动态Prompt过长({tokenized_length}>{max_prompt_length})，回退原始Prompt\n")
                #     f.write(f"  动态Prompt片段: {dynamic_prompt[:100]}...\n")
                #     f.write(f"  原始Prompt片段: {prompt[:100]}...\n")
                #     f.write(f"  动态token数: {tokenized_length}, 原始token数: {original_tokenized_length}\n")
            
            dynamic_prompts.append(dynamic_prompt)
            
        
        # # 保存前检查动态prompts（保持原逻辑）
        # valid_count = sum(1 for dp in dynamic_prompts if dp.strip())
        # with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        #     f.write(f"[DEBUG][Step {self.global_steps}] 动态prompts生成完成，总数: {len(dynamic_prompts)}, 有效数: {valid_count}, 回退数: {fallback_count}\n")
        #     f.write(f"[DEBUG] 有效dynamic_prompts示例（前1条）:\n")
        #     for i, dp in enumerate(dynamic_prompts[:1]):
        #         f.write(f"--- 有效dynamic_prompt {i+1} (长度: {len(dp)}) ---\n")
        #         f.write(dp + "\n" if dp else "(空prompt)\n")
        
        
        # try:
        #     with open(SAVE_PATH, "a", encoding="utf-8") as f:
        #         f.write(f"===== Step {self.global_steps} ==== (总数: {len(dynamic_prompts)}, 有效数: {valid_count}, 回退数: {fallback_count})\n")
        #         for i, dp in enumerate(dynamic_prompts):
        #             f.write(f"--- Prompt {i+1} ---\n")
        #             f.write(dp + "\n" if dp else "(空prompt)\n")
        #             f.write("="*50 + "\n")
        #     with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        #         f.write(f"[DEBUG] 动态prompts已保存至 {SAVE_PATH}\n")
        # except Exception as e:
        #     with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        #         f.write(f"[ERROR] 保存动态prompts失败: {str(e)}\n")
        #     print(f"[ERROR] 保存动态prompts失败: {str(e)}")
        
        # with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        #     f.write(f"[END] 动态prompt生成 - Step {self.global_steps}\n{'='*80}\n")
        
        return dynamic_prompts


        


    
   
