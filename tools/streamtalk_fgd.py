#!/usr/bin/env python3

# Copyright (c) 2021, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Official EMAGE/PantoMatrix protocol, adapted for StreamTalk in 2026.

"""Official full-clip EMAGE FGD with persistent AESK/GT state."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence


AESK_SHA256 = "5cd9566b24264f34d44003b3de62cdfd50aa85b7cdde2d369214599023c40f55"


class ContractError(RuntimeError):
    """An input does not satisfy the released EMAGE FGD protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TEMPORAL_MULTIPLE = 32
FID_EPSILON = 1e-6


def per_clip_truncated_lengths(lengths: Sequence[int], multiple: int = TEMPORAL_MULTIPLE) -> tuple[int, ...]:
    """Expose the official boundary-preserving truncation contract for audits/tests."""
    if multiple <= 0 or not lengths:
        raise ValueError("multiple and lengths must be nonempty/positive")
    result=tuple(int(length)//multiple*multiple for length in lengths)
    if any(value<=0 for value in result):
        raise ContractError(f"every FGD clip needs at least {multiple} frames")
    return result


def official_axis_angle_to_rotation_6d(axis_angle: Any, torch_module: Any) -> Any:
    """Exact PantoMatrix quaternion conversion (including small-angle branch)."""

    rotations = axis_angle.reshape(*axis_angle.shape[:-1], axis_angle.shape[-1] // 3, 3)
    angles = torch_module.norm(rotations, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles.abs() < 1e-6
    ratio = torch_module.empty_like(angles)
    ratio[~small] = torch_module.sin(half_angles[~small]) / angles[~small]
    ratio[small] = 0.5 - (angles[small] * angles[small]) / 48
    quaternion = torch_module.cat((torch_module.cos(half_angles), rotations * ratio), dim=-1)
    real, i, j, k = torch_module.unbind(quaternion, -1)
    two_s = 2.0 / (quaternion * quaternion).sum(-1)
    matrices = torch_module.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * real),
            two_s * (i * k + j * real), two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k), two_s * (j * k - i * real),
            two_s * (i * k - j * real), two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ), dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return matrices[..., :2, :].clone().reshape(*axis_angle.shape[:-1], -1)


def _rows(array: Any, name: str) -> Any:
    import numpy as np

    value = np.asarray(array)
    if value.ndim < 2 or value.dtype.kind != "f" or value.shape[-1] <= 0:
        raise ContractError(f"{name} must be a floating latent array")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains NaN/Inf")
    return value.reshape(-1, value.shape[-1])


def _statistics(features: Any, name: str) -> tuple[int, Any, Any]:
    import numpy as np

    samples = _rows(features, name)
    if samples.shape[0] < 2:
        raise ContractError(f"{name} needs at least two rows")
    # Deliberately no dtype override: official map2latent mean remains float32,
    # while np.cov promotes covariance to float64.
    mean = np.mean(samples, axis=0)
    covariance = np.atleast_2d(np.cov(samples, rowvar=False))
    return int(samples.shape[0]), mean, covariance


def emage_fgd_from_statistics(
    generated: tuple[int, Any, Any],
    real: tuple[int, Any, Any],
    epsilon: float = FID_EPSILON,
) -> float:
    import numpy as np
    from scipy.linalg import sqrtm

    gen_count, gen_mean, gen_cov = generated
    real_count, real_mean, real_cov = real
    if real_count < 2 or gen_count < 2 or real_mean.shape != gen_mean.shape:
        raise ContractError("incompatible FGD features")
    offset = np.eye(real_mean.size, dtype=np.float64) * float(epsilon)
    # Preserve official argument order: cov(prediction) @ cov(ground_truth).
    root = sqrtm((gen_cov + offset).dot(real_cov + offset))
    if np.iscomplexobj(root):
        root = root.real
    difference = real_mean - gen_mean
    score = float(
        difference.dot(difference) + np.trace(real_cov) + np.trace(gen_cov)
        - 2.0 * np.trace(root)
    )
    if not math.isfinite(score):
        raise ContractError("official EMAGE FGD is non-finite")
    return score


def emage_fgd(generated_rows: Any, real_rows: Any, epsilon: float = FID_EPSILON) -> float:
    """Official order is generated first, real second."""
    return emage_fgd_from_statistics(
        _statistics(generated_rows, "generated features"),
        _statistics(real_rows, "real features"),
        epsilon,
    )


class PersistentOfficialFGD:
    """Encode each GT clip once, preserving per-file 32-frame truncation."""

    def __init__(
        self,
        encoder: Callable[[Any], Any],
        ground_truth_clips: Sequence[Any],
        batch_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("FGD batch_size must be positive")
        self.encoder = encoder
        self.batch_size = int(batch_size)
        self.ground_truth_clips = self._truncate(ground_truth_clips, "ground truth")
        self.ground_truth_rows, self.rows_per_clip = self._encode(
            self.ground_truth_clips, "ground truth"
        )
        self.ground_truth_statistics = _statistics(
            self.ground_truth_rows, "ground-truth features"
        )

    @staticmethod
    def _truncate(clips: Sequence[Any], label: str) -> tuple[Any, ...]:
        import numpy as np

        values = []
        dimension = None
        lengths=per_clip_truncated_lengths([len(raw) for raw in clips])
        for index, (raw,frames) in enumerate(zip(clips,lengths)):
            clip = np.asarray(raw, dtype=np.float32)
            if clip.ndim != 2 or not np.isfinite(clip).all():
                raise ContractError(f"{label} clip {index} is invalid")
            if dimension is None:
                dimension = clip.shape[1]
            elif dimension != clip.shape[1]:
                raise ContractError(f"{label} clips have inconsistent dimensions")
            values.append(clip[:frames])
        if not values:
            raise ContractError("FGD requires at least one clip")
        return tuple(values)

    def _encode(self, clips: Sequence[Any], label: str) -> tuple[Any, tuple[int, ...]]:
        import numpy as np

        if self.batch_size==1:
            rows=[]; counts=[]
            for index,clip in enumerate(clips):
                encoded=self.encoder(clip[None,...])
                if hasattr(encoded,"detach"):
                    encoded=encoded.detach().cpu().numpy()
                encoded=_rows(encoded,f"{label} latent {index}")
                rows.append(encoded); counts.append(encoded.shape[0])
            return np.concatenate(rows,axis=0),tuple(counts)

        # Batch only equal-length clips, then restore exact clip order and
        # boundaries before statistics.  batch_size=1 retains the released
        # EMAGE oracle path and is the production default until A/B acceptance.
        grouped: dict[int, list[int]] = {}
        for index, clip in enumerate(clips):
            grouped.setdefault(len(clip), []).append(index)
        rows: list[Any | None] = [None] * len(clips)
        for length in sorted(grouped):
            indices = grouped[length]
            for start in range(0, len(indices), self.batch_size):
                selected = indices[start : start + self.batch_size]
                payload = np.stack([clips[index] for index in selected], axis=0)
                encoded = self.encoder(payload)
                if hasattr(encoded, "detach"):
                    encoded = encoded.detach().cpu().numpy()
                encoded = np.asarray(encoded)
                if encoded.ndim < 3 or encoded.shape[0] != len(selected):
                    raise ContractError(
                        f"{label} batched encoder output has invalid shape {encoded.shape}"
                    )
                for batch_row, clip_index in enumerate(selected):
                    rows[clip_index] = _rows(
                        encoded[batch_row], f"{label} latent {clip_index}"
                    )
        if any(value is None for value in rows):
            raise AssertionError("FGD batching lost a clip")
        completed = [value for value in rows if value is not None]
        return np.concatenate(completed, axis=0), tuple(
            int(value.shape[0]) for value in completed
        )

    def evaluate(self, prediction_clips: Sequence[Any]) -> dict[str, Any]:
        predictions = self._truncate(prediction_clips, "prediction")
        if len(predictions) != len(self.ground_truth_clips):
            raise ContractError("prediction/GT clip counts differ")
        for index, (prediction, reference) in enumerate(zip(predictions, self.ground_truth_clips)):
            if prediction.shape != reference.shape:
                raise ContractError(f"prediction/GT clip {index} shapes differ")
        prediction_rows, counts = self._encode(predictions, "prediction")
        return self.evaluate_encoded(prediction_rows, counts)

    def encode_predictions(
        self, prediction_clips: Sequence[Any]
    ) -> tuple[Any, tuple[int, ...]]:
        """Validate/truncate/encode one candidate while retaining clip row extents."""
        predictions = self._truncate(prediction_clips, "prediction")
        if len(predictions) != len(self.ground_truth_clips):
            raise ContractError("prediction/GT clip counts differ")
        for index, (prediction, reference) in enumerate(
            zip(predictions, self.ground_truth_clips)
        ):
            if prediction.shape != reference.shape:
                raise ContractError(f"prediction/GT clip {index} shapes differ")
        return self._encode(predictions, "prediction")

    def evaluate_encoded(
        self, prediction_rows: Any, counts: Sequence[int]
    ) -> dict[str, Any]:
        """Score already encoded rows, enabling a zero-copy scope subset."""
        prediction_rows = _rows(prediction_rows, "prediction latent rows")
        counts = tuple(int(value) for value in counts)
        if counts != self.rows_per_clip:
            raise ContractError("prediction/GT latent rows per clip differ")
        return {
            # Official EMAGE calls frechet_distance(prediction, ground_truth).
            # sqrtm(A@B) can have tiny order-dependent floating differences, so
            # keep this argument order explicit rather than assuming symmetry.
            "emage_fgd": emage_fgd_from_statistics(
                _statistics(prediction_rows, "prediction features"),
                self.ground_truth_statistics,
            ),
            "clip_count": len(counts),
            "frame_count": sum(
                clip.shape[0] for clip in self.ground_truth_clips
            ),
            "latent_row_count": int(prediction_rows.shape[0]),
            "latent_dimension": int(prediction_rows.shape[1]),
            "temporal_multiple": TEMPORAL_MULTIPLE,
            "epsilon": FID_EPSILON,
        }


def load_persistent_aesk(device: Any, torch_module: Any, asset_dir: Path) -> tuple[Any, Callable[[Any], Any]]:
    checkpoint = asset_dir / "AESKConv_240_100.bin"
    if sha256_file(checkpoint) != AESK_SHA256:
        raise ContractError(f"AESK SHA-256 gate failed: {checkpoint}")
    from Scripts.FM.EMAGE_VAE.test_vae import GetVAE

    model = GetVAE(device).requires_grad_(False).eval()

    def encode(clips: Any) -> Any:
        tensor = torch_module.as_tensor(clips, dtype=torch_module.float32, device=device)
        with torch_module.inference_mode():
            rotation = official_axis_angle_to_rotation_6d(tensor[..., :165], torch_module)
            return model.map2latent(rotation[..., :330])

    return model, encode
