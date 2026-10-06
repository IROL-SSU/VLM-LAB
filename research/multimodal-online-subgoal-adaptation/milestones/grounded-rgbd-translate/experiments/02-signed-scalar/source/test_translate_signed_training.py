"""Unit tests for the signed-displacement model and safe interval loss."""
import unittest

import torch

from test_translate_subgoal_model import synthetic_points
from translate_signed_subgoal_model import (
    DIRECTION_NAMES,
    SignedTranslateSubgoalModel,
    decode,
    safe_interval_loss,
    safe_signed_bounds,
    signed_exact_target,
)


class SignedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_signed_output_and_joint_gradient(self):
        torch.manual_seed(7)
        model = SignedTranslateSubgoalModel()
        output = model.from_features(torch.rand(2, 1536), synthetic_points())
        self.assertEqual(output["signed_displacement_m"].shape, (2,))
        self.assertTrue((output["signed_displacement_m"].abs() < .25).all())
        direction = torch.tensor([0, 1])
        lower, upper, _ = safe_signed_bounds(
            direction,
            torch.tensor([.05, .06]),
            torch.tensor([.10, .12]),
        )
        loss, _ = safe_interval_loss(output, lower, upper)
        loss.backward()
        self.assertGreater(float(model.displacement_head.weight.grad.abs().sum()), 0)
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.geometry_encoder.parameters()
        ))

    def test_signed_bounds_apply_direction_and_margin(self):
        direction = torch.tensor([0, 1])
        minimum = torch.tensor([.05, .05])
        maximum = torch.tensor([.09, .09])
        lower, upper, margin = safe_signed_bounds(direction, minimum, maximum)
        torch.testing.assert_close(lower, torch.tensor([-.08, .06]), atol=2e-5, rtol=0)
        torch.testing.assert_close(upper, torch.tensor([-.06, .08]), atol=2e-5, rtol=0)
        torch.testing.assert_close(margin, torch.tensor([.01, .01]), atol=2e-5, rtol=0)

    def test_narrow_interval_adapts_margin(self):
        lower, upper, margin = safe_signed_bounds(
            torch.tensor([1]),
            torch.tensor([.050]),
            torch.tensor([.052]),
        )
        self.assertAlmostEqual(float(margin), .0004975, places=6)
        self.assertGreater(float(upper - lower), 0)

    def test_interval_loss_is_zero_inside_and_penalizes_wrong_sign(self):
        lower = torch.tensor([-.08, .06])
        upper = torch.tensor([-.06, .08])
        loss, violation = safe_interval_loss(
            {"signed_displacement_m": torch.tensor([-.07, .07])}, lower, upper
        )
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(violation.sum()), 0)
        wrong_loss, wrong_violation = safe_interval_loss(
            {"signed_displacement_m": torch.tensor([.07, -.07])}, lower, upper
        )
        self.assertGreater(float(wrong_loss), 0)
        self.assertTrue((wrong_violation > .1).all())

    def test_decode_and_exact_target(self):
        output = {"signed_displacement_m": torch.tensor([-.06, .08])}
        direction, distance, signed = decode(output)
        self.assertEqual(DIRECTION_NAMES[direction[0]], "LEFT")
        self.assertEqual(DIRECTION_NAMES[direction[1]], "RIGHT")
        torch.testing.assert_close(distance, torch.tensor([.06, .08]))
        torch.testing.assert_close(signed_exact_target(direction, distance), signed)


if __name__ == "__main__":
    unittest.main()
