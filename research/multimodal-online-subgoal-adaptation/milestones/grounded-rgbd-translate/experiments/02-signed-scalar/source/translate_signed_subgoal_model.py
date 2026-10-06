"""LEFT/RIGHT RGB-D sub-goal model with one signed displacement output.

Negative output means LEFT and positive output means RIGHT.  The model keeps
the observation-only RGB and point-cloud encoders from the registered baseline,
but couples direction and distance in a single scalar prediction.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from translate_subgoal_model import (
    DISTANCE_SCALE_M,
    INPUT_FILES,
    FrozenRGBEncoder,
    GeometryEncoder,
    observation_batch,
    read_json,
)


DIRECTION_NAMES = ("LEFT", "RIGHT")
LEFT_INDEX = 0
RIGHT_INDEX = 1


class SignedTranslateSubgoalModel(nn.Module):
    """Predict a shelf-X displacement in ``[-DISTANCE_SCALE_M, +scale]``."""

    def __init__(self, rgb_weights=None):
        super().__init__()
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
        self.displacement_head = nn.Linear(64, 1)

    def from_features(self, rgb_features, points):
        geometry = self.geometry_encoder(points)
        fused = self.fusion(torch.cat([
            self.rgb_projection(rgb_features),
            self.geo_projection(geometry),
        ], dim=1))
        signed = torch.tanh(self.displacement_head(fused)).squeeze(1) * DISTANCE_SCALE_M
        return {"signed_displacement_m": signed}

    def forward(self, image, masks, points):
        return self.from_features(self.rgb_encoder(image, masks), points)


def signed_exact_target(direction, distance_m):
    """Convert LEFT/RIGHT plus magnitude into a signed shelf-X displacement."""
    sign = torch.where(
        direction == LEFT_INDEX,
        distance_m.new_tensor(-1.0),
        distance_m.new_tensor(1.0),
    )
    return sign * distance_m


def safe_signed_bounds(
    direction,
    minimum_distance_m,
    maximum_clear_distance_m,
    max_edge_margin_m=0.01,
):
    """Build a non-empty signed interval strictly inside the feasible range.

    ``minimum_distance_m`` is the first corridor-clearing translation and
    ``maximum_clear_distance_m`` is the collision/boundary limit from dataset
    generation.  Up to ``max_edge_margin_m`` is removed from both interval
    edges; narrow feasible ranges use one quarter of their available slack.
    """
    if not 0 <= max_edge_margin_m < DISTANCE_SCALE_M:
        raise ValueError("max_edge_margin_m must be in [0, distance scale)")
    if not (
        direction.shape == minimum_distance_m.shape == maximum_clear_distance_m.shape
    ):
        raise ValueError("Direction and distance targets must have identical shapes")
    if not torch.all((direction == LEFT_INDEX) | (direction == RIGHT_INDEX)):
        raise ValueError("Signed model only supports LEFT and RIGHT")
    if not torch.isfinite(minimum_distance_m).all() or not torch.isfinite(maximum_clear_distance_m).all():
        raise ValueError("Safe-distance targets must be finite")

    # tanh cannot attain the exact scale and the geometry checker uses a strict
    # distance < clearance-limit comparison, so stay a small epsilon inside.
    scale_limit = minimum_distance_m.new_tensor(DISTANCE_SCALE_M - 1e-5)
    maximum = torch.minimum(maximum_clear_distance_m - 1e-5, scale_limit)
    slack = maximum - minimum_distance_m
    if not torch.all(slack > 0):
        raise ValueError("Every sample needs a feasible interval above its minimum distance")
    edge_margin = torch.minimum(
        slack * 0.25,
        minimum_distance_m.new_full(minimum_distance_m.shape, max_edge_margin_m),
    )
    lower_magnitude = minimum_distance_m + edge_margin
    upper_magnitude = maximum - edge_margin

    signed_lower = torch.where(direction == LEFT_INDEX, -upper_magnitude, lower_magnitude)
    signed_upper = torch.where(direction == LEFT_INDEX, -lower_magnitude, upper_magnitude)
    if not torch.all(signed_lower < signed_upper):
        raise AssertionError("Safe signed interval must be non-empty")
    return signed_lower, signed_upper, edge_margin


def safe_interval_loss(output, signed_lower_m, signed_upper_m):
    """Smooth loss equal to zero inside the signed feasible interval."""
    prediction = output["signed_displacement_m"]
    below = (signed_lower_m - prediction).clamp_min(0)
    above = (prediction - signed_upper_m).clamp_min(0)
    violation = below + above
    loss = F.smooth_l1_loss(
        violation / DISTANCE_SCALE_M,
        torch.zeros_like(violation),
        beta=0.02,
    )
    return loss, violation


def decode(output):
    """Return LEFT/RIGHT index, unsigned distance, and signed displacement."""
    signed = output["signed_displacement_m"]
    direction = torch.where(signed < 0, LEFT_INDEX, RIGHT_INDEX).long()
    return direction, signed.abs(), signed
