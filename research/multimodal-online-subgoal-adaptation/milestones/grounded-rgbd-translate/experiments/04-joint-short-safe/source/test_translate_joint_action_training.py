"""Unit tests for mirrored joint direction-distance action training."""
import unittest

import torch

from test_translate_subgoal_model import synthetic_points
from translate_joint_action_model import (
    JointTranslateActionModel,
    SHELF_X_CENTER_M,
    acceptable_set_loss,
    decode,
    mirror_action_mask,
    mirror_points,
    preferred_action_mask,
    preferred_acceptable_set_loss,
)


class JointActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_shape_decode_and_gradient(self):
        torch.manual_seed(7)
        distances = [.03, .05, .08]
        model = JointTranslateActionModel(distances)
        output = model.from_features(torch.rand(2, 1536), synthetic_points())
        self.assertEqual(output["action_logits"].shape, (2, 6))
        acceptable = torch.tensor([
            [True, False, False, False, False, False],
            [False, False, False, False, True, False],
        ])
        loss, set_nll, ranking = acceptable_set_loss(output, acceptable)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(set_nll))
        self.assertTrue(torch.isfinite(ranking))
        loss.backward()
        self.assertGreater(float(model.action_head.weight.grad.abs().sum()), 0)
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.geometry_encoder.parameters()
        ))

    def test_decode_uses_joint_class(self):
        distances = [.03, .05, .08]
        logits = torch.full((2, 6), -10.)
        logits[0, 1] = 10.
        logits[1, 5] = 10.
        direction, distance, confidence, action = decode(
            {"action_logits": logits}, distances
        )
        torch.testing.assert_close(direction, torch.tensor([0, 1]))
        torch.testing.assert_close(distance, torch.tensor([.05, .08]))
        torch.testing.assert_close(action, torch.tensor([1, 5]))
        self.assertTrue((confidence > .99).all())

    def test_set_loss_accepts_any_safe_class(self):
        acceptable = torch.tensor([[False, True, True, False]])
        safe_logits = torch.tensor([[-8., 8., -8., -8.]])
        other_safe_logits = torch.tensor([[-8., -8., 8., -8.]])
        unsafe_logits = torch.tensor([[8., -8., -8., -8.]])
        safe_loss = acceptable_set_loss(
            {"action_logits": safe_logits}, acceptable
        )[0]
        other_safe_loss = acceptable_set_loss(
            {"action_logits": other_safe_logits}, acceptable
        )[0]
        unsafe_loss = acceptable_set_loss(
            {"action_logits": unsafe_logits}, acceptable
        )[0]
        self.assertLess(float(safe_loss), .01)
        self.assertLess(float(other_safe_loss), .01)
        self.assertGreater(float(unsafe_loss), 10.)

    def test_mirror_swaps_direction_blocks(self):
        mask = torch.tensor([[True, False, False, False, True, False]])
        expected = torch.tensor([[False, True, False, True, False, False]])
        torch.testing.assert_close(mirror_action_mask(mask), expected)
        torch.testing.assert_close(
            mirror_action_mask(mirror_action_mask(mask)), mask
        )

    def test_preferred_action_is_shortest_robust_direction(self):
        # LEFT is safe from 50 through 100 mm and RIGHT from 80 through 120 mm.
        # A 10 mm inward margin gives LEFT 60 mm versus RIGHT 90 mm.
        distances = [.05, .06, .07, .08, .09, .10, .11, .12]
        acceptable = torch.tensor([[
            True, True, True, True, True, True, False, False,
            False, False, False, True, True, True, True, True,
        ]])
        preferred = preferred_action_mask(acceptable, distances, .01)
        expected = torch.zeros_like(acceptable)
        expected[0, 1] = True
        torch.testing.assert_close(preferred, expected)

    def test_narrow_interval_adapts_preferred_margin(self):
        distances = [.05, .055, .06]
        acceptable = torch.tensor([[
            True, True, False,
            False, False, False,
        ]])
        preferred = preferred_action_mask(acceptable, distances, .01)
        expected = torch.tensor([[
            True, False, False,
            False, False, False,
        ]])
        torch.testing.assert_close(preferred, expected)

    def test_preferred_loss_favors_preferred_but_keeps_safe_set(self):
        acceptable = torch.tensor([[False, True, True, False]])
        preferred = torch.tensor([[False, True, False, False]])
        preferred_logits = torch.tensor([[-8., 8., -8., -8.]])
        alternate_safe_logits = torch.tensor([[-8., -8., 8., -8.]])
        preferred_loss = preferred_acceptable_set_loss(
            {"action_logits": preferred_logits}, acceptable, preferred
        )[0]
        alternate_loss = preferred_acceptable_set_loss(
            {"action_logits": alternate_safe_logits}, acceptable, preferred
        )[0]
        unsafe_loss = preferred_acceptable_set_loss(
            {"action_logits": torch.tensor([[8., -8., -8., -8.]])},
            acceptable,
            preferred,
        )[0]
        self.assertLess(float(preferred_loss), float(alternate_loss))
        self.assertLess(float(alternate_loss), float(unsafe_loss))

    def test_mirror_points_is_shelf_centered_and_reversible(self):
        points = synthetic_points()
        mirrored = mirror_points(points)
        torch.testing.assert_close(
            mirrored[..., 0], 2 * SHELF_X_CENTER_M - points[..., 0]
        )
        torch.testing.assert_close(mirrored[..., 1:], points[..., 1:])
        torch.testing.assert_close(mirror_points(mirrored), points)

    def test_invalid_acceptable_set_rejected(self):
        with self.assertRaises(ValueError):
            acceptable_set_loss(
                {"action_logits": torch.zeros(1, 4)},
                torch.zeros(1, 4, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
