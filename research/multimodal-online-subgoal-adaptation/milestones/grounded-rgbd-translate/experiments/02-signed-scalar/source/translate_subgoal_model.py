"""Observed RGB-D -> frozen ResNet18 + trainable PointNet-style sub-goal model.

No simulator object bounds, asset identity, goal images or labels enter the
encoder. XYZ remains in a shared metric shelf frame (no per-object centering).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from generate_translate_rgbd_pilot import backproject
from translate_pilot_geometry import BOUNDS, DIRECTIONS

DIRECTION_NAMES = tuple(DIRECTIONS)
IMAGE_HW = (480, 640)
DISTANCE_SCALE_M = 0.25
POINTS_PER_ROLE = 512
SCENE_POINTS = 1024
INPUT_FILES = ("rgb.png", "depth_z_m.npy", "target_mask.png", "blocker_mask.png", "calibration.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_observation(folder, seed=20261001):
    """Only the five declared sensor/grounding files are read by this function."""
    folder = Path(folder)
    rgb = np.array(Image.open(folder / "rgb.png").convert("RGB"), dtype=np.float32) / 255
    depth = np.load(folder / "depth_z_m.npy", allow_pickle=False)
    masks = np.stack([np.array(Image.open(folder / name)) > 0 for name in ("target_mask.png", "blocker_mask.png")])
    calib = read_json(folder / "calibration.json")
    if rgb.shape[:2] != depth.shape or masks.shape[1:] != depth.shape:
        raise ValueError("RGB, depth and masks must be pixel aligned")
    if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(masks[0] & masks[1]):
        raise ValueError("Invalid depth or overlapping visible role masks")
    xyz = backproject(depth, calib)
    crop = ((depth > 0) & (xyz[..., 0] >= BOUNDS[0] - .02) & (xyz[..., 0] <= BOUNDS[2] + .02)
            & (xyz[..., 1] >= BOUNDS[1] - .02) & (xyz[..., 1] <= BOUNDS[3] + .02)
            & (xyz[..., 2] >= -.01) & (xyz[..., 2] <= .37))
    cloud = np.concatenate([xyz[crop], masks[:, crop].T.astype(np.float32)], axis=1)
    rng = np.random.default_rng(seed)
    selected = []
    for pool, count in ((np.flatnonzero(cloud[:, 3] > .5), POINTS_PER_ROLE),
                        (np.flatnonzero(cloud[:, 4] > .5), POINTS_PER_ROLE),
                        (np.arange(len(cloud)), SCENE_POINTS)):
        if not len(pool):
            raise ValueError(f"Missing valid depth for a role / scene: {folder}")
        selected.append(rng.choice(pool, size=count, replace=len(pool) < count))
    image = torch.from_numpy(rgb).permute(2, 0, 1)
    image = F.interpolate(image[None], size=IMAGE_HW, mode="bilinear", align_corners=False, antialias=True)[0]
    mean = image.new_tensor([.485, .456, .406])[:, None, None]
    std = image.new_tensor([.229, .224, .225])[:, None, None]
    role_masks = F.interpolate(torch.from_numpy(masks.astype(np.float32))[None], size=IMAGE_HW, mode="area")[0]
    return {"image": (image - mean) / std, "masks": role_masks,
            "points": torch.from_numpy(cloud[np.concatenate(selected)].astype(np.float32))}


def observation_batch(dataset, rows, seed=20261001):
    # The same observation-only RNG seed is used per scene; no scene ID / label
    # is encoded as a numeric input or as a sampling-seed feature.
    items = [load_observation(Path(dataset) / row["sample_dir"], seed) for row in rows]
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def masked_mean(features, mask):
    weights = mask.to(features.dtype)
    return (features * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)


class FrozenRGBEncoder(nn.Module):
    output_dim = 512 * 3

    def __init__(self, weights_path=None):
        super().__init__()
        backbone = resnet18(weights=None)
        if weights_path is not None:
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        self.trunk = nn.Sequential(*list(backbone.children())[:-2])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen BatchNorm buffers must not change on this five-scene dataset.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image, masks):
        spatial = self.trunk(image)
        soft_masks = F.adaptive_avg_pool2d(masks, spatial.shape[-2:])
        features = spatial.flatten(2).transpose(1, 2)
        groups = [masked_mean(features, mask.flatten(1)) for mask in soft_masks.unbind(1)]
        groups.append(features.mean(1))
        return torch.cat(groups, dim=1)


class GeometryEncoder(nn.Module):
    output_dim = 3 * (128 * 2 + 3) + 3

    def __init__(self):
        super().__init__()
        self.register_buffer("origin", torch.tensor([.28, 0., 0.]))
        self.register_buffer("scale", torch.tensor([.92, .32, .37]))
        self.point_mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                       nn.Linear(128, 128), nn.ReLU())

    def forward(self, points):
        xyz = (points[..., :3] - self.origin) / self.scale
        features = self.point_mlp(torch.cat([xyz, points[..., 3:]], dim=-1))
        masks = [points[..., 3] > .5, points[..., 4] > .5,
                 torch.ones_like(points[..., 3], dtype=torch.bool)]
        groups, centers = [], []
        for mask in masks:
            if not mask.any(1).all():
                raise ValueError("Each batch element needs visible target and blocker points")
            center = masked_mean(xyz, mask)
            centers.append(center)
            groups.extend([features.masked_fill(~mask[..., None], -torch.inf).amax(1),
                           masked_mean(features, mask), center])
        # Explicit observed relative position; no GT bounds or object centers.
        groups.append(centers[0] - centers[1])
        return torch.cat(groups, dim=1)


class TranslateSubgoalModel(nn.Module):
    def __init__(self, rgb_weights=None):
        super().__init__()
        self.rgb_encoder = FrozenRGBEncoder(rgb_weights)
        self.geometry_encoder = GeometryEncoder()
        self.rgb_projection = nn.Sequential(nn.LayerNorm(FrozenRGBEncoder.output_dim), nn.Linear(FrozenRGBEncoder.output_dim, 128), nn.ReLU())
        self.geo_projection = nn.Sequential(nn.LayerNorm(GeometryEncoder.output_dim), nn.Linear(GeometryEncoder.output_dim, 128), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.direction_head = nn.Linear(64, len(DIRECTION_NAMES))
        self.distance_head = nn.Linear(64, len(DIRECTION_NAMES))

    def from_features(self, rgb_features, points):
        geometry = self.geometry_encoder(points)
        fused = self.fusion(torch.cat([self.rgb_projection(rgb_features), self.geo_projection(geometry)], dim=1))
        # One distance per direction, trained only at the ground-truth direction.
        return {"logits": self.direction_head(fused),
                "distances_m": torch.sigmoid(self.distance_head(fused)) * DISTANCE_SCALE_M}

    def forward(self, image, masks, points):
        return self.from_features(self.rgb_encoder(image, masks), points)


def supervised_loss(output, direction, distance_m):
    classification = F.cross_entropy(output["logits"], direction)
    selected = output["distances_m"].gather(1, direction[:, None]).squeeze(1)
    regression = F.smooth_l1_loss(selected / DISTANCE_SCALE_M, distance_m / DISTANCE_SCALE_M, beta=.02)
    return classification + 5 * regression, classification, regression


def decode(output):
    probabilities = output["logits"].softmax(-1)
    direction = probabilities.argmax(-1)
    distance = output["distances_m"].gather(1, direction[:, None]).squeeze(1)
    confidence = probabilities.gather(1, direction[:, None]).squeeze(1)
    return direction, distance, confidence
