"""CPU tests for Main3: formal FedEPPU without model EMA."""
import contextlib
import io
import unittest
import torch
import numpy as np
from torch import nn
import FedEPPU3 as engine
import main3 as main
from pu_loss import pu_pairwise_ranking_loss


class TinyDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 8)
        self.classifier = nn.Linear(8, 1)
        self.ranking_head = nn.Linear(8, 1)

    def forward_dual(self, x):
        h = self.shared(x.flatten(1)).tanh()
        return self.classifier(h), self.ranking_head(h), h

    def forward(self, x):
        return self.forward_dual(x)[0]


class StratifiedPUTest(unittest.TestCase):
    def setUp(self):
        self.previous_device = engine.device
        engine.device = torch.device('cpu')
        torch.manual_seed(42)

    def tearDown(self):
        engine.device = self.previous_device

    def test_main3_has_no_model_ema_api_or_setting(self):
        self.assertFalse(hasattr(engine, "_update_model_ema"))
        self.assertTrue(hasattr(engine, "ProxyCheckpointInference"))
        self.assertFalse(hasattr(engine, "EMASelectionQuantileInference"))

    def test_sampler_preserves_pool_membership_and_counts(self):
        p, u = torch.tensor([0, 1, 2]), torch.arange(3, 24)
        idx = engine.sample_pu_indices(p, u, 256)
        self.assertEqual(idx.numel(), 256)
        self.assertTrue(torch.isin(idx[:64], p).all())
        self.assertTrue(torch.isin(idx[64:], u).all())
        with self.assertRaises(ValueError):
            engine.sample_pu_indices(p, u[:0], 256)

    def test_all_experiment_datasets_use_the_unified_dual_head_model(self):
        for dataset_name in (
            'fashion_mnist', 'cifar10', 'chest', 'kolektor_sdd2'
        ):
            cfg = main._CONFIGS[dataset_name]
            model = main.build_model(cfg)
            self.assertIsInstance(model, engine.BetterCifarCNN)
            self.assertTrue(hasattr(model, 'forward_dual'))
            self.assertEqual(model.features[0].in_channels, cfg.input_channels)

    def test_known_priors_are_used_without_shrinkage(self):
        raw = np.asarray([0.2, 0.8], dtype=np.float64)
        state = engine.KnownPriorState.build(raw, np.asarray([10, 90]))
        self.assertTrue(np.array_equal(state.effective, raw))

    def test_continuous_expected_negative_support(self):
        self.assertEqual(engine.linear_ranking_support(.5, 192), 1.0)
        self.assertAlmostEqual(
            engine.linear_ranking_support(.9986, 192), .2688, places=6
        )
        self.assertAlmostEqual(
            engine.linear_ranking_support(.9995, 192), .096, places=6
        )

        model = TinyDual()
        logits, ranking, features = model.forward_dual(torch.rand(256, 1, 2, 2))
        labels = torch.tensor([1] * 64 + [-1] * 192)
        diagnostics = {}
        _, effective_scale = engine.ClassificationRankingPUObjective.compute(
            logits, ranking, features, labels, .9995,
            diagnostics=diagnostics,
        )
        self.assertAlmostEqual(diagnostics['ranking_support'], .096, places=6)
        self.assertEqual(float(effective_scale.item()), 1.0)
        self.assertEqual(diagnostics['local_support_factor'], 1.0)
        self.assertNotIn('gradient_balance_scale', diagnostics)

    def test_support_aware_aggregation_weights(self):
        base = np.asarray([.4923, .0077, .5])
        support = np.asarray([.27, 1., 1.])
        actual = engine.ComplementMassAggregation.support_aware_weights(base, support)
        expected = base * support
        expected /= expected.sum()
        self.assertTrue(np.allclose(actual, expected))
        self.assertAlmostEqual(float(actual.sum()), 1.0)
        self.assertTrue(np.allclose(
            engine.ComplementMassAggregation.support_aware_weights(base, np.ones(3)),
            base / base.sum(),
        ))
        self.assertTrue(np.allclose(
            engine.ComplementMassAggregation.support_aware_weights(base, np.zeros(3)),
            base / base.sum(),
        ))

    def test_diagnostics_preserve_ranking_loss_and_complement_weights(self):
        scores_a = torch.randn(16, requires_grad=True)
        scores_b = scores_a.detach().clone().requires_grad_(True)
        labels = torch.tensor([1] * 4 + [-1] * 12)
        expected = pu_pairwise_ranking_loss(scores_a, labels, .4)
        diagnostics = {}
        actual = pu_pairwise_ranking_loss(scores_b, labels, .4, diagnostics)
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(scores_a.grad, scores_b.grad))
        self.assertIn('ranking_raw_risk', diagnostics)
        for counts in ([10, 20, 70], [1, 1, 1], [20, 30]):
            counts = np.asarray(counts, dtype=np.float64)
            complement_mass = np.minimum(counts, counts.sum() - counts)
            expected_weights = complement_mass / complement_mass.sum()
            self.assertTrue(np.allclose(
                expected_weights, engine.client_weights(counts)
            ))
        weights = engine.client_weights(np.asarray([364, 3786, 44826]))
        self.assertTrue(np.allclose(weights, np.asarray([364, 3786, 4150]) / 8300))

    def test_dual_loss_and_diagnostics_at_extreme_priors(self):
        x, labels = torch.rand(24, 1, 2, 2), torch.tensor([1] * 3 + [-1] * 21)
        for prior in (.0009, .5, .9995):
            model = TinyDual()
            model, diag = engine.local_train(model, x, labels, prior, .0015, 16, 2, 1, False)
            self.assertEqual(diag['steps'], 2)
            self.assertEqual(diag['positive_per_batch'], 4)
            self.assertEqual(diag['ranking_head_present'], 1)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertTrue(0 <= diag['nnpu_correction_fraction'] <= 1)
            self.assertEqual(diag['actual_steps_per_round'], 2)
            self.assertEqual(diag['actual_steps_per_epoch'], 2)
            self.assertEqual(diag['ranking_scale_mean'], 1.0)
            self.assertEqual(diag['local_support_factor_mean'], 1.0)
            self.assertNotIn('gradient_balance_scale_mean', diag)
            self.assertIn('ranking_raw_risk_mean', diag)
            self.assertIn('ranking_abs_correction_fraction', diag)
            if prior == .9995:
                self.assertGreater(diag['ranking_active_fraction'], 0)
                self.assertGreater(diag['ranking_support_mean'], 0)
                self.assertLess(diag['ranking_support_mean'], 1)
            if prior == .5:
                self.assertGreater(diag['ranking_active_fraction'], 0)

    def test_federated_selection_and_history(self):
        x, labels = torch.rand(24, 1, 2, 2), torch.tensor([1] * 6 + [-1] * 18)
        proxy, target = torch.rand(16, 1, 2, 2), torch.tensor([1, -1] * 8)
        with contextlib.redirect_stdout(io.StringIO()):
            model, history, acc = engine.train_fedavg_proposed(
                client_data=[(x, labels)], x_test=proxy, t_test=target,
                proxy_x=proxy, proxy_t=target, priors=[.5], anchor_model=TinyDual(),
                rounds=2, n_clients=1,
                batch_size=16, max_steps_per_epoch=1, local_epochs=1, use_amp=False,
            )
        self.assertIn(history['selected']['round'], (1, 2))
        self.assertEqual(len(history['client_diag']), 2)
        self.assertEqual(len(history['proxy_auc']), 2)
        self.assertIn('effective_aggregation_weight', history['client_diag'][0][0])
        self.assertTrue(0 <= acc <= 1)


if __name__ == '__main__':
    unittest.main()


