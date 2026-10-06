"""Joint LEFT/RIGHT and distance-action classifier for RGB-D translation.

Each class is one executable action such as LEFT 50 mm or RIGHT 85 mm.  A
scene may supervise several acceptable classes, while inference still selects
one class.  Horizontal mirroring reflects shelf-X around the shelf center and
swaps the LEFT/RIGHT action blocks.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from translate_pilot_geometry import BOUNDS
from translate_subgoal_model import FrozenRGBEncoder, GeometryEncoder


DIRECTION_NAMES = ("LEFT", "RIGHT")
LEFT_INDEX = 0
RIGHT_INDEX = 1
SHELF_X_CENTER_M = (BOUNDS[0] + BOUNDS[2]) / 2


class JointTranslateActionModel(nn.Module):
    """Classify a single joint direction-distance action."""

    def __init__(self, action_distances_m, rgb_weights=None):
        super().__init__()
        distances = torch.as_tensor(action_distances_m, dtype=torch.float32)
        if distances.ndim != 1 or not len(distances):
            raise ValueError("action_distances_m must be a non-empty vector")
        if not torch.all(torch.isfinite(distances)) or not torch.all(distances > 0):
            raise ValueError("Action distances must be finite and positive")
        if len(distances) > 1 and not torch.all(distances[1:] > distances[:-1]):
            raise ValueError("Action distances must be strictly increasing")
        self.register_buffer("action_distances_m", distances)

        self.rgb_encoder = FrozenRGBEncoder(rgb_weights)
        self.geometry_encoder = GeometryEncoder()
        self.rgb_projection = nn.Sequential(
            nn.LayerNorm(FrozenRGBEncoder.output_dim),
            nn.Linear(FrozenRGBEncoder.output_dim, 128),
            nn.ReLU(),
        )
        self.geo_projection = nn.Sequential(
            nn.LayerNorm(GeometryEncoder.output_dim),
            nn.Linear(GeometryEncoder.output_dim, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, 2 * len(distances))

    @property
    def action_count(self):
        return 2 * len(self.action_distances_m)

    def from_features(self, rgb_features, points):
        geometry = self.geometry_encoder(points)
        fused = self.fusion(torch.cat([
            self.rgb_projection(rgb_features),
            self.geo_projection(geometry),
        ], dim=1))
        return {"action_logits": self.action_head(fused)}

    def forward(self, image, masks, points):
        return self.from_features(self.rgb_encoder(image, masks), points)


def acceptable_set_loss(
    output,
    acceptable_mask,
    ranking_weight=0.25,
    ranking_margin=0.2,
):
    """Reward total probability on acceptable actions and a safe argmax.

    The first term is ``-log(sum(p[a] for a in acceptable_actions))``.  A
    small ranking term additionally asks the strongest acceptable logit to
    beat every unacceptable logit, because inference selects one argmax.
    """
    logits = output["action_logits"]
    if acceptable_mask.shape != logits.shape or acceptable_mask.dtype != torch.bool:
        raise ValueError("acceptable_mask must be a bool tensor matching action logits")
    if not acceptable_mask.any(dim=1).all():
        raise ValueError("Every sample needs at least one acceptable action")

    log_probabilities = F.log_softmax(logits, dim=1)
    acceptable_log_mass = torch.logsumexp(
        log_probabilities.masked_fill(~acceptable_mask, -torch.inf), dim=1
    )
    set_nll = -acceptable_log_mass.mean()

    best_acceptable = logits.masked_fill(~acceptable_mask, -torch.inf).amax(dim=1)
    has_unacceptable = (~acceptable_mask).any(dim=1)
    best_unacceptable = logits.masked_fill(acceptable_mask, -torch.inf).amax(dim=1)
    ranking_per_sample = torch.where(
        has_unacceptable,
        (best_unacceptable - best_acceptable + ranking_margin).clamp_min(0),
        torch.zeros_like(best_acceptable),
    )
    ranking = ranking_per_sample.mean()
    return set_nll + ranking_weight * ranking, set_nll, ranking


def preferred_action_mask(
    acceptable_mask,
    action_distances_m,
    max_edge_margin_m=0.01,
):
    """Choose the shortest robust action while retaining all safe labels.

    For each direction, move inward from both edges of its discrete safe
    interval by ``min(max_edge_margin_m, interval_width / 4)``.  The first
    remaining distance is that direction's robust candidate.  Across LEFT and
    RIGHT, the shorter robust candidate is preferred; exact ties are retained.
    """
    distances = torch.as_tensor(
        action_distances_m,
        dtype=torch.float32,
        device=acceptable_mask.device,
    )
    if acceptable_mask.ndim != 2 or acceptable_mask.dtype != torch.bool:
        raise ValueError("acceptable_mask must be a two-dimensional bool tensor")
    if acceptable_mask.shape[1] != 2 * len(distances):
        raise ValueError("Action catalog does not match acceptable mask width")
    if not acceptable_mask.any(dim=1).all():
        raise ValueError("Every sample needs at least one acceptable action")
    if not 0 <= max_edge_margin_m < 0.25:
        raise ValueError("max_edge_margin_m must be in [0, 0.25)")

    preferred = torch.zeros_like(acceptable_mask)
    magnitude_count = len(distances)
    for batch_index in range(len(acceptable_mask)):
        directional_candidates = []
        blocks = acceptable_mask[batch_index].reshape(2, magnitude_count)
        for direction_index, block in enumerate(blocks):
            safe_indices = block.nonzero(as_tuple=False).flatten()
            if not len(safe_indices):
                continue
            safe_distances = distances[safe_indices]
            width = safe_distances[-1] - safe_distances[0]
            margin = min(max_edge_margin_m, float(width) / 4)
            robust = safe_indices[
                (safe_distances >= safe_distances[0] + margin - 1e-8)
                & (safe_distances <= safe_distances[-1] - margin + 1e-8)
            ]
            if not len(robust):
                robust = safe_indices
            action_index = direction_index * magnitude_count + int(robust[0])
            directional_candidates.append((float(distances[robust[0]]), action_index))
        best_distance = min(distance for distance, _ in directional_candidates)
        for distance, action_index in directional_candidates:
            if abs(distance - best_distance) <= 1e-8:
                preferred[batch_index, action_index] = True
    if not preferred.any(dim=1).all():
        raise AssertionError("Every sample must receive a preferred action")
    if (preferred & ~acceptable_mask).any():
        raise AssertionError("Preferred actions must be acceptable")
    return preferred


def preferred_acceptable_set_loss(
    output,
    acceptable_mask,
    preferred_mask,
    preferred_weight=0.2,
    ranking_weight=0.25,
    ranking_margin=0.2,
):
    """Keep all safe actions valid while softly favoring the robust shortest."""
    if preferred_mask.shape != acceptable_mask.shape or preferred_mask.dtype != torch.bool:
        raise ValueError("preferred_mask must be a bool tensor matching acceptable_mask")
    if not preferred_mask.any(dim=1).all() or (preferred_mask & ~acceptable_mask).any():
        raise ValueError("Preferred actions must be a non-empty subset of acceptable actions")
    base, set_nll, ranking = acceptable_set_loss(
        output,
        acceptable_mask,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
    )
    log_probabilities = F.log_softmax(output["action_logits"], dim=1)
    preferred_nll = -torch.logsumexp(
        log_probabilities.masked_fill(~preferred_mask, -torch.inf), dim=1
    ).mean()
    return base + preferred_weight * preferred_nll, set_nll, ranking, preferred_nll


def decode(output, action_distances_m):
    """Decode one action index into LEFT/RIGHT, distance, and confidence."""
    logits = output["action_logits"]
    distances = torch.as_tensor(
        action_distances_m, dtype=logits.dtype, device=logits.device
    )
    if logits.shape[1] != 2 * len(distances):
        raise ValueError("Action catalog does not match model output width")
    probabilities = logits.softmax(dim=1)
    action_index = probabilities.argmax(dim=1)
    magnitude_index = action_index.remainder(len(distances))
    direction = torch.div(action_index, len(distances), rounding_mode="floor")
    distance = distances[magnitude_index]
    confidence = probabilities.gather(1, action_index[:, None]).squeeze(1)
    return direction, distance, confidence, action_index


def mirror_points(points):
    """Reflect observed shelf-frame points around the shelf's X center."""
    if points.shape[-1] != 5:
        raise ValueError("Expected XYZ plus target/blocker role flags")
    mirrored = points.clone()
    mirrored[..., 0] = 2 * SHELF_X_CENTER_M - mirrored[..., 0]
    return mirrored


def mirror_action_mask(mask):
    """Swap LEFT and RIGHT blocks while retaining each distance class."""
    if mask.ndim != 2 or mask.shape[1] % 2:
        raise ValueError("Expected a [batch, 2 * magnitude_count] mask")
    return mask.reshape(mask.shape[0], 2, -1).flip(1).reshape_as(mask)
