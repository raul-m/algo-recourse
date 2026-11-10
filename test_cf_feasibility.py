"""Scientific checks for the two-arm CF experiment, including upstream parity."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cf_feasibility_experiment import (
    AdultPolicy, BlackBox, Config, DEFAULT_OUT, ModelApproxCF,
    candidate_metrics, load_data, objective, state_hash,
)


def reference_definition(filename, name, namespace):
    path = DEFAULT_OUT / "upstream" / filename
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    assert len(selected) == 1
    # Extract only the inspected model/loss definition, not imports, downloads,
    # command-line parsing, or the repository's top-level training commands.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.a, cls.meta = load_data(DEFAULT_OUT)
        cls.policy = AdultPolicy(cls.meta)
        cls.x = torch.from_numpy(cls.a["x"][cls.a["train_indices"][:16]])

    def test_upstream_architecture_and_initialization_match(self):
        ref = reference_definition("vae_model.py", "CF_VAE", dict(torch=torch, nn=nn, F=F))
        class Metadata:
            def get_data_params(_self):
                return None, None, [list(range(s, e)) for s, e in self.policy.blocks]
        torch.manual_seed(101)
        original = ref(29, 10, Metadata())
        torch.manual_seed(101)
        adapted = ModelApproxCF()
        self.assertEqual(state_hash(original), state_hash(adapted))
        original.eval(); adapted.eval()
        targets = torch.arange(len(self.x)) % 2
        torch.manual_seed(109)
        source = original(self.x, targets)
        torch.manual_seed(109)
        raw, mean, variance = adapted(self.x, targets, 50)
        torch.testing.assert_close(raw, torch.stack(source["x_pred"]))
        torch.testing.assert_close(mean, source["em"])
        torch.testing.assert_close(variance, source["ev"])

    def test_original_loss_matches_upstream_including_quirks(self):
        torch.manual_seed(29)
        classifier = BlackBox().eval()
        classifier.requires_grad_(False)
        reference = reference_definition("unary-const-generative-cf.py", "compute_loss",
                     dict(torch=torch, F=F, pred_model=classifier, cuda=torch.device("cpu")))
        raw = torch.rand(3, len(self.x), 29)
        mean, variance = torch.rand(len(self.x), 10), torch.rand(len(self.x), 10) + 0.5
        target = torch.arange(len(self.x)) % 2
        cfg = Config()
        class Info:
            encoded_start_cat = 2
            encoded_categorical_feature_indexes = [list(range(s, e)) for s, e in self.policy.blocks]
        output = dict(em=mean, ev=variance, z=[], x_pred=list(raw.unbind()), mc_samples=3)
        weights = {f["start"]: [f["min"], f["max"]] for f in self.policy.numeric}
        with redirect_stdout(io.StringIO()):
            expected = reference(Info(), output, self.x, target, weights, cfg.validity_weight, cfg.margin)
        expected = expected + cfg.age_weight * F.relu(self.x[:, 0] - raw[..., 0]).mean()
        actual, _ = objective(raw, mean, variance, self.x, target, classifier, self.policy, cfg, False)
        torch.testing.assert_close(actual, expected.squeeze(), rtol=1e-6, atol=1e-5)

    def test_projection_guarantees_each_rule_and_is_idempotent(self):
        raw = torch.rand(4, len(self.x), 29)
        raw[..., 0] = 0
        for index in self.policy.banned.values():
            raw[..., index] = 10
        for hard in (False, True):
            projected = self.policy.project(raw, self.x, hard=hard)
            torch.testing.assert_close(projected, self.policy.project(projected, self.x, hard=hard))
            self.assertTrue(torch.all(projected[..., 0] >= self.x[:, 0]))
            for name in self.policy.fixed_names:
                f = self.policy.by_name[name]; s, e = f["start"], f["end"]
                torch.testing.assert_close(projected[..., s:e], self.x[None, :, s:e].expand(4, -1, -1))
            for index in self.policy.banned.values():
                self.assertTrue(torch.all(projected[..., index] == 0))
            for s, e in self.policy.blocks:
                torch.testing.assert_close(projected[..., s:e].sum(-1), torch.ones(4, len(self.x)))
                if hard:
                    self.assertTrue(torch.all((projected[..., s:e] == 0) | (projected[..., s:e] == 1)))

    def test_gradients_flow_through_mutable_but_not_copied_features(self):
        raw = torch.full((2, len(self.x), 29), 0.5, requires_grad=True)
        projected = self.policy.project(raw, self.x)
        weights = torch.arange(1, 30, dtype=raw.dtype)
        (projected * weights).sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad[..., 1].abs().sum()), 0)
        for name in self.policy.fixed_names:
            f = self.policy.by_name[name]
            self.assertTrue(torch.all(raw.grad[..., f["start"]:f["end"]] == 0))
        for index in self.policy.banned.values():
            self.assertTrue(torch.all(raw.grad[..., index] == 0))

    def test_empty_target_class_is_finite_and_classifier_frozen(self):
        classifier = BlackBox().eval()
        classifier.requires_grad_(False)
        before = state_hash(classifier)
        for label in (0, 1):
            raw = torch.rand(2, len(self.x), 29, requires_grad=True)
            mean, variance = torch.zeros(len(self.x), 10), torch.ones(len(self.x), 10)
            for extended in (False, True):
                loss, stats = objective(raw, mean, variance, self.x, torch.full((len(self.x),), label), classifier, self.policy, Config(), extended)
                self.assertTrue(torch.isfinite(loss))
                loss.backward(retain_graph=True)
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertGreater(float(stats["age_penalty"].detach()), 0)
        self.assertEqual(before, state_hash(classifier))
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))

    def test_decoding_does_not_secretly_enforce_feasibility(self):
        raw = self.x[None].clone()
        raw[..., 0] = 0
        for name, index in self.policy.banned.items():
            f = self.policy.by_name[name]
            raw[..., f["start"]:f["end"]] = 0
            raw[..., index] = 1
        decoded = self.policy.decode(raw)
        m = candidate_metrics(decoded.numpy(), self.x.numpy(), np.zeros(len(self.x)), np.ones((1, len(self.x))), self.policy)
        self.assertTrue(m["valid"].all())
        self.assertTrue(m["viol_workclass_unknown"].all())
        self.assertFalse(m["valid_feasible"].any())

    def test_saved_split_is_exact_and_disjoint(self):
        source = json.loads(Path("relax-assets/adult/data/config.json").read_text())
        np.testing.assert_array_equal(self.a["train_indices"], source["train_indices"])
        np.testing.assert_array_equal(self.a["test_indices"], source["test_indices"])
        self.assertEqual(len(np.intersect1d(self.a["train_indices"], self.a["test_indices"])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
