import math
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
saved = sys.argv
sys.argv = [sys.argv[0], "--cfg_file", str(ROOT / "configs/pretrain.yaml")]
try:
    from lib.config import cfg
    from lib.networks.diffusion.flow_matching_evolution import FlowMatchingEvolution
finally:
    sys.argv = saved

from lib.protocol import protocol_metadata
from lib.rl.geometry import sample_geometry, chart_logprob, fourier_basis
from lib.train.states import mixed_states, terminal_credit
from lib.train.rewards.region_reward import contour_metrics


def circle(count=128):
    angle = torch.arange(count) * (2 * math.pi / count)
    return torch.stack((20 + 10 * angle.cos(), 20 + 10 * angle.sin()), -1)[None]


class ProtocolTests(unittest.TestCase):
    def test_gdpo_cannot_silently_train_supervised_loss(self):
        model = FlowMatchingEvolution.__new__(FlowMatchingEvolution)
        torch.nn.Module.__init__(model)
        previous = cfg.training_stage
        try:
            cfg.training_stage = "gdpo"
            with self.assertRaisesRegex(RuntimeError, "policy objective"):
                model({}, None, {})
        finally:
            cfg.training_stage = previous

    def test_geometry_probability_replay_and_gradient(self):
        for local in (False, True):
            for repeated in (False, True):
                state = circle()
                if repeated:
                    state[:, 10:18] = state[:, 9:10]
                mean = torch.randn_like(state).requires_grad_()
                action, old, component = sample_geometry(mean, state, .2, local=local)
                replay = chart_logprob(action, mean, component)
                self.assertLess(float((old - replay).abs().max()), .002)
                replay.sum().backward()
                self.assertTrue(torch.isfinite(mean.grad).all())
                self.assertGreater(float(mean.grad.abs().sum()), 0)

    def test_fourier_is_orthonormal(self):
        basis = fourier_basis(128, 8, "cpu", torch.float64)
        self.assertLess(float((basis.T @ basis - torch.eye(8)).abs().max()), 1e-12)

    def test_credit_mean_sign_and_zero_terminal(self):
        reward = torch.tensor([[.1], [.4], [.7]])
        quality = torch.tensor([[[0.], [0.]], [[.1], [.8]], [[.9], [.1]]])
        advantage, weights, allocated = terminal_credit(reward, quality)
        torch.testing.assert_close(allocated.mean(1), advantage)
        torch.testing.assert_close(weights.mean(1), torch.ones_like(advantage))
        self.assertTrue((allocated * advantage[:, None] >= 0).all())
        self.assertEqual(float(terminal_credit(torch.ones_like(reward), quality)[2].abs().max()), 0)

    def test_common_delta_cancels(self):
        reward = torch.rand(8, 4)
        quality = torch.rand(8, 2, 4)
        original = terminal_credit(reward, quality)[0]
        shifted = terminal_credit(reward - torch.rand(1, 4), quality)[0]
        torch.testing.assert_close(original, shifted)

    def test_mixed_states_cover_continuous_middle_and_phase(self):
        torch.manual_seed(7)
        initial = circle().expand(10000, -1, -1).clone()
        target = initial + 3
        current, phase, kind = mixed_states(initial, target)
        proportions = torch.bincount(kind, minlength=3) / 10000
        torch.testing.assert_close(proportions, torch.tensor([.5, .2, .3]), atol=.02, rtol=0)
        self.assertTrue((phase[kind == 0] == 0).all())
        self.assertTrue((phase[kind == 2] == 1).all())
        self.assertEqual(set(phase[kind == 1].tolist()), {0., 1.})
        residual = target - current
        torch.testing.assert_close(current + residual, target)
        alpha = ((current[kind == 1] - initial[kind == 1]) / 3).flatten()
        self.assertGreater(torch.unique(alpha.round(decimals=3)).numel(), 100)

    def test_multiscale_quality_identity_empty_and_coordinate_scale(self):
        poly = circle()
        same = contour_metrics(poly, poly, 128, 128)
        self.assertEqual(float(same["quality"]), 1.)
        shifted = contour_metrics(poly + 1, poly, 128, 128)
        self.assertLessEqual(float(shifted["nsd_1px"]), float(shifted["nsd_8px"]))
        scaled = contour_metrics((poly + 1) * 4, poly * 4, 128, 128, coord_scale=1)
        torch.testing.assert_close(shifted["quality"], scaled["quality"])
        empty = contour_metrics(poly[:0], poly[:0], 128, 128)
        self.assertEqual(empty["quality"].numel(), 0)

    def test_only_one_block_and_no_dead_implementations(self):
        import ast
        definitions = []
        for path in (ROOT / "lib").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            definitions.extend((path, node.name) for node in ast.walk(tree)
                               if isinstance(node, ast.ClassDef) and "DiTBlock" in node.name)
        self.assertEqual([name for _, name in definitions], ["MainlineDiTBlock"])
        self.assertFalse((ROOT / "lib/networks/diffusion/ha_smoe.py").exists())
        self.assertFalse((ROOT / "lib/rl/fourier.py").exists())
        self.assertFalse((ROOT / "tools/train_rl.py").exists())
        self.assertEqual(protocol_metadata()["total_nfe"], 8)

    def test_exact_two_actions_and_true_ab2(self):
        class Cache(torch.nn.Module):
            def set_kv_cache(self, context):
                pass
            def clear_kv_cache(self):
                pass
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit_layers = torch.nn.ModuleList([Cache()])
                self.local_proj = torch.nn.Identity()
            def global_compressor(self, feature):
                return feature.flatten(2).transpose(1, 2)
        model = FlowMatchingEvolution.__new__(FlowMatchingEvolution)
        torch.nn.Module.__init__(model)
        model.denoiser = Dummy()
        model._disp_norm_enabled = False
        calls = []
        def velocity(feature, sampled, x, time, owner, phase):
            calls.append((float(time[0]), float(phase[0])))
            return time[:, None, None].expand_as(x), x.new_zeros(())
        model.predict_velocity = velocity
        current = torch.zeros(1, 128, 2)
        result = model.rollout(torch.zeros(1, 1, 4, 4), current, torch.tensor([0]),
                               [torch.zeros_like(current), torch.zeros_like(current)])
        self.assertEqual(len(calls), 8)
        self.assertEqual([p for _, p in calls], [0.] * 4 + [1.] * 4)
        torch.testing.assert_close(result["py"], torch.full_like(current, .9375))
        self.assertEqual(len(result["actions"]), 2)
        with self.assertRaises(ValueError):
            model.rollout(torch.zeros(1, 1, 4, 4), current, torch.tensor([0]), [current])


if __name__ == "__main__":
    unittest.main()
