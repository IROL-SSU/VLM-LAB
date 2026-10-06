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
