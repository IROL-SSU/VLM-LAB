"""Safety checks for fixed splits and full-coverage minibatches."""
import unittest

import torch

from train_translate_dataset import minibatches, validate_splits, prediction_metrics
from translate_subgoal_model import DIRECTION_NAMES


class DatasetTrainingTests(unittest.TestCase):
    def fixtures(self):
        rows = [{"scene_id": str(i), "sample_dir": f"samples/{i}"} for i in range(6)]
        names = ("train", "validation", "test")
        manifests = {name: {"samples": rows[i * 2:i * 2 + 2]} for i, name in enumerate(names)}
        ids = {name: [r["scene_id"] for r in m["samples"]] for name, m in manifests.items()}
        return {"samples": rows}, manifests, ids

    def test_valid_splits(self):
        result = validate_splits(*self.fixtures())
        self.assertEqual({k: len(v) for k, v in result.items()}, {"train": 2, "validation": 2, "test": 2})

    def test_overlap_rejected(self):
        manifest, splits, ids = self.fixtures()
        splits["test"]["samples"][0] = splits["train"]["samples"][0]
        with self.assertRaises(ValueError):
            validate_splits(manifest, splits, ids)

    def test_mismatched_row_rejected(self):
        manifest, splits, ids = self.fixtures()
        splits["test"]["samples"][0] = {"scene_id": "4", "sample_dir": "samples/0"}
        with self.assertRaises(ValueError):
            validate_splits(manifest, splits, ids)

    def test_minibatch_coverage_reproducibility(self):
        first = torch.cat(minibatches(101, 32, torch.Generator().manual_seed(3), True))
        second = torch.cat(minibatches(101, 32, torch.Generator().manual_seed(3), True))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(sorted(first.tolist()), list(range(101)))
        self.assertEqual([len(x) for x in minibatches(101, 32)], [32, 32, 32, 5])

    def test_metric_uses_predicted_distance_not_gt_head(self):
        left, right = DIRECTION_NAMES.index("LEFT"), DIRECTION_NAMES.index("RIGHT")
        logits = torch.zeros(1, 8)
        logits[0, right] = 10
        distances = torch.full((1, 8), .2)
        distances[0, left] = .05
        metrics = prediction_metrics({"logits": logits, "distances_m": distances}, torch.tensor([left]), torch.tensor([.05]))
        self.assertEqual(metrics["direction_accuracy"], 0)
        self.assertAlmostEqual(metrics["distance_mae_mm"], 150, places=3)
        self.assertEqual(metrics["gt_direction_head_distance_mae_mm"], 0)


if __name__ == "__main__":
    unittest.main()
