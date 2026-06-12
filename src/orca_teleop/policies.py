"""Generic policy adapters for ORCA teleop/sim control.

The runtime contract is deliberately small: a policy receives a LeRobot-style
observation mapping and returns an action that can be converted to
``OrcaJointPositions``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from orca_core import OrcaJointPositions

_MISSING = object()
_JOINT_NAME_ALIASES = {
    "thumb_pip": "thumb_cmc",
    "thumb_cmc": "thumb_pip",
}


@runtime_checkable
class GenericPolicy(Protocol):
    """Minimal interface shared by LeRobot policies and local test policies."""

    def select_action(self, observation: Mapping[str, Any]) -> Any: ...


class ResettablePolicy(GenericPolicy, Protocol):
    """Optional extension for policies with recurrent/chunked internal state."""

    def reset(self) -> None: ...


def reset_policy(policy: GenericPolicy) -> None:
    """Reset policy state when the object exposes a reset hook."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def action_to_joint_positions(action: Any, joint_ids: list[str]) -> OrcaJointPositions:
    """Convert a policy action into physical-degree ORCA joint positions.

    Supported policy outputs:
    - ``OrcaJointPositions``: returned as-is.
    - Mapping with an ``"action"`` entry: recurses into that value.
    - Mapping keyed by every joint id: values are used directly.
    - Tensor/array/list with shape ``(n_joints,)``, ``(1, n_joints)``, or
      ``(1, n_action_steps, n_joints)``. Chunked outputs use the first action.
    """
    if isinstance(action, OrcaJointPositions):
        return action

    if isinstance(action, Mapping):
        if "action" in action:
            return action_to_joint_positions(action["action"], joint_ids)
        values = {joint_id: _resolve_named_value(joint_id, action) for joint_id in joint_ids}
        if all(value is not _MISSING for value in values.values()):
            return OrcaJointPositions.from_dict(
                {joint_id: float(value) for joint_id, value in values.items()}
            )
        missing = [joint_id for joint_id, value in values.items() if value is _MISSING]
        raise ValueError(f"Policy action mapping is missing joint keys: {missing}")

    arr = _as_numpy(action)
    if arr.ndim == 0:
        raise ValueError("Policy action must contain one value per joint, got a scalar.")
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for chunked action, got shape {arr.shape}.")
        arr = arr[0, 0]
    elif arr.ndim == 2:
        arr = arr[0] if arr.shape[0] == 1 else arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"Unsupported policy action shape: {arr.shape}")

    if arr.shape[0] != len(joint_ids):
        raise ValueError(
            f"Policy action has {arr.shape[0]} values, but sink expects {len(joint_ids)} joints."
        )
    values = dict(zip(joint_ids, arr.astype(float).tolist(), strict=True))
    return OrcaJointPositions.from_dict(values)


@dataclass
class LeRobotPolicyAdapter:
    """Policy-agnostic LeRobot inference wrapper.

    The checkpoint's ``config.json`` declares its policy type, so the concrete
    policy class (ACT, Diffusion Policy, pi0, ...) is resolved through
    LeRobot's policy factory rather than hardcoded here.

    This class intentionally imports LeRobot lazily so the rest of ``orca_teleop``
    remains usable in environments that only run teleop or tests.
    """

    policy: Any
    preprocess: Any
    postprocess: Any
    dataset_features: Mapping[str, Any]
    device: Any
    build_inference_frame: Any | None = None
    make_robot_action: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        policy_path: str,
        dataset_repo_id: str,
        *,
        device: str = "cpu",
        dataset_root: str | None = None,
    ) -> LeRobotPolicyAdapter:
        """Load any LeRobot policy and its normalization metadata from a checkpoint."""
        try:
            import torch

            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            except ImportError:
                from lerobot.datasets import LeRobotDatasetMetadata
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            try:
                from lerobot.policies.utils import build_inference_frame, make_robot_action
            except ImportError:
                build_inference_frame = None
                make_robot_action = None
        except ImportError as exc:
            raise ImportError(
                "LeRobot policy inference requires the 'lerobot' package. "
                "Install this project with the learning extra or install lerobot directly."
            ) from exc

        torch_device = torch.device(device)
        policy_config = PreTrainedConfig.from_pretrained(policy_path)
        policy = get_policy_class(policy_config.type).from_pretrained(policy_path)
        if hasattr(policy, "config"):
            policy.config.device = str(torch_device)
        policy.to(torch_device)
        policy.eval()

        metadata_kwargs = {"repo_id": dataset_repo_id}
        if dataset_root is not None:
            metadata_kwargs["root"] = dataset_root
        dataset_metadata = LeRobotDatasetMetadata(**metadata_kwargs)
        try:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                dataset_stats=dataset_metadata.stats,
            )
        except TypeError:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                dataset_metadata.stats,
            )

        return cls(
            policy=policy,
            preprocess=preprocess,
            postprocess=postprocess,
            dataset_features=dataset_metadata.features,
            device=torch_device,
            build_inference_frame=build_inference_frame,
            make_robot_action=make_robot_action,
        )

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def select_action(self, observation: Mapping[str, Any]) -> Any:
        """Run one inference step from a LeRobot-style observation mapping."""
        import torch

        observation = self._align_observation(observation)
        try:
            if self.build_inference_frame is None:
                raise RuntimeError("LeRobot build_inference_frame utility unavailable.")
            frame = self.build_inference_frame(
                observation=dict(observation),
                ds_features=self.dataset_features,
                device=self.device,
            )
        except Exception:
            frame = _tensorize_observation(observation, self.device)

        with torch.inference_mode():
            batch = self.preprocess(frame)
            action = self.policy.select_action(batch)
            action = self.postprocess(action)

        named_action = self._named_action(action)
        if named_action is not None:
            return named_action
        if self.make_robot_action is None:
            return action
        try:
            return self.make_robot_action(action, self.dataset_features)
        except Exception:
            return action

    def _align_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt a simulator observation to the trained dataset feature shape."""
        aligned = dict(observation)

        state_feature = self.dataset_features.get("observation.state")
        if state_feature is not None and "observation.state" in aligned:
            expected_shape = state_feature.get("shape")
            if expected_shape:
                expected_width = int(expected_shape[0])
                state = np.asarray(aligned["observation.state"], dtype=np.float32)
                state_names = state_feature.get("names")
                source_names = aligned.get("observation.state.names")
                if state.ndim == 1 and state_names and source_names:
                    source = dict(zip(source_names, state.tolist(), strict=True))
                    aligned["observation.state"] = np.asarray(
                        _ordered_named_values(state_names, source),
                        dtype=np.float32,
                    )
                    return_aligned_width = aligned["observation.state"].shape[0]
                    if return_aligned_width == expected_width:
                        state = aligned["observation.state"]
                if state.ndim == 1 and state.shape[0] < expected_width:
                    padded = np.zeros(expected_width, dtype=np.float32)
                    padded[-state.shape[0] :] = state
                    aligned["observation.state"] = padded

        image_keys = [key for key in self.dataset_features if key.startswith("observation.images.")]
        available_image = next(
            (value for key, value in aligned.items() if key.startswith("observation.images.")),
            None,
        )
        for key in image_keys:
            if key not in aligned:
                aligned[key] = (
                    available_image
                    if available_image is not None
                    else _blank_image_like_feature(self.dataset_features[key])
                )

        return aligned

    def _named_action(self, action: Any) -> dict[str, float] | None:
        action_feature = self.dataset_features.get("action")
        if action_feature is None:
            return None
        action_names = action_feature.get("names")
        if not action_names:
            return None

        arr = _as_numpy(action)
        if arr.ndim == 3:
            arr = arr[0, 0]
        elif arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim != 1:
            return None
        if arr.shape[0] != len(action_names):
            return None
        return dict(zip(action_names, arr.astype(float).tolist(), strict=True))


# Deprecated alias kept for backwards compatibility: the adapter is no longer
# ACT-specific since the policy class is resolved from the checkpoint config.
LeRobotACTPolicyAdapter = LeRobotPolicyAdapter


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _tensorize_observation(observation: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    batch: dict[str, Any] = {}
    for key, value in observation.items():
        if key.endswith(".names"):
            continue
        if key == "task":
            batch[key] = value if isinstance(value, list) else [value]
            continue

        arr = np.asarray(value)
        tensor = torch.as_tensor(arr, device=device)
        if key.startswith("observation.images."):
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float()
            if tensor.numel() and tensor.max() > 1:
                tensor = tensor / 255.0
        else:
            tensor = tensor.float()

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor.unsqueeze(0)

    return batch


def _blank_image_like_feature(feature: Mapping[str, Any]) -> np.ndarray:
    shape = feature.get("shape", (1, 1, 3))
    if len(shape) != 3:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height, width, channels = (int(dim) for dim in shape)
    return np.zeros((height, width, channels), dtype=np.uint8)


def _resolve_named_value(name: str, values: Mapping[str, Any]) -> Any:
    for candidate in _joint_name_candidates(name):
        if candidate in values:
            return values[candidate]
    return _MISSING


def _ordered_named_values(names: list[str], values: Mapping[str, Any]) -> list[float]:
    ordered = []
    for name in names:
        value = _resolve_named_value(name, values)
        ordered.append(0.0 if value is _MISSING else float(value))
    return ordered


def _joint_name_candidates(name: str) -> list[str]:
    candidates = [name]
    side_prefix = ""
    base_name = name
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            side_prefix = prefix
            base_name = name.removeprefix(prefix)
            candidates.append(base_name)
            break
    else:
        candidates.extend([f"left_{name}", f"right_{name}"])

    alias = _JOINT_NAME_ALIASES.get(base_name)
    if alias is not None:
        candidates.append(alias)
        candidates.extend([f"left_{alias}", f"right_{alias}"])
        if side_prefix:
            candidates.append(f"{side_prefix}{alias}")

    return list(dict.fromkeys(candidates))
