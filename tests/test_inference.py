"""CPU inference checks; optionally exercise the full release via an environment variable.

Run: python -m unittest discover -s tests -p 'test_inference.py'
Full release: EOSFM_RELEASE_CHECKPOINT=release/eosfm.pth <same command>
"""
import gc
import os
from pathlib import Path
import tempfile
import unittest

import timm
import torch
from torch import nn

import eosfm
from eosfm import EosFM, SmartBandAdapter
from eosfm.encoder import WNConv2d


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'specialists.pth'
        torch.manual_seed(625)
        specialist = timm.create_model('convnextv2_atto', in_chans=3, features_only=True, pretrained=False)
        cls.state = specialist.state_dict()
        torch.save([(3, cls.state, {'backbone': 'timm_convnextv2_atto', 'encoder_name': name,
                                  'bands': ['R', 'G', 'B']}) for name in ('first', 'second')], cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, **kwargs):
        return EosFM(input_bands={'optical': ['R', 'G', 'B']}, input_size=32,
                     encoder_weights=self.path, **kwargs)

    def test_public_import_and_raw_features(self):
        self.assertIs(EosFM, eosfm.EosFM)
        model = self.model(normalize_features=False).eval()
        self.assertEqual(model.out_channels, [80, 160, 320, 640])
        self.assertTrue(all(isinstance(m, nn.Identity) for m in model.feature_fusion))
        image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            maps = model(image)
            dictionary_maps = model({'optical': image.unsqueeze(2)})
            expected = model.encoders[0](image)
        for actual, dictionary, expected_level in zip(maps, dictionary_maps, expected):
            torch.testing.assert_close(actual, torch.cat([expected_level, expected_level], 1), rtol=0, atol=0)
            torch.testing.assert_close(actual, dictionary, rtol=0, atol=0)
        pruned = model.export_pruned_model([1])
        self.assertFalse(pruned.training)
        self.assertEqual(pruned.encoder_names, ['second'])
        with torch.no_grad():
            for actual, expected_level in zip(pruned(image), expected):
                torch.testing.assert_close(actual, expected_level, rtol=0, atol=0)

    def test_frozen_specialists_and_bn_modes(self):
        model = self.model()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        bn = model.feature_normalizers[0][0]
        model.train()
        self.assertTrue(bn.training)
        self.assertTrue(all(not encoder.training for encoder in model.encoders))
        image = torch.randn(2, 3, 32, 32)
        before = bn.running_mean.clone()
        with torch.no_grad():
            model(image)
        self.assertFalse(torch.equal(before, bn.running_mean))
        model.eval()
        before = bn.running_mean.clone()
        with torch.no_grad():
            model(image)
        torch.testing.assert_close(before, bn.running_mean, rtol=0, atol=0)

    def test_pruning_preserves_finetuned_bn_and_projection_state(self):
        model = self.model(projection_layer=True, finetune_encoders=True)
        image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            next(model.encoders[1].parameters()).add_(0.01)
            model(image)  # Fit nontrivial BN running statistics.
        model.eval()
        pruned = model.export_pruned_model([1])
        with torch.no_grad():
            full = model(image)
            compact = pruned(image)
        for expected, actual in zip(full, compact):
            torch.testing.assert_close(actual, expected[:, expected.shape[1] // 2:], rtol=0, atol=0)
        self.assertNotEqual(next(model.encoders[1].parameters()).data_ptr(),
                            next(pruned.encoders[0].parameters()).data_ptr())

    def test_adapter_retains_expression_and_missing_cloud_policy(self):
        adapter = SmartBandAdapter(['R', 'G', 'B'], ['VV', 'VH', 'VV-VH', 'CLD'])
        image = torch.tensor([[[[5.]], [[2.]], [[1.]]]])
        # SAR fallback cycles R,G, then computes their difference. Missing CLD
        # follows the historical optical fallback cursor (third channel here).
        torch.testing.assert_close(adapter.adapt(image).flatten(), torch.tensor([5., 2., 3., 1.]))
        adapter = SmartBandAdapter(['B02', 'B03'], ['CLD', 'B2'])
        torch.testing.assert_close(adapter.adapt(image[:, :2]).flatten(), torch.tensor([5., 5.]))

    def test_weight_normalized_fusion_and_selection_delay(self):
        model = self.model(normalize_features=False, feature_fusion='conv1x1', max_encoders=1,
                           encoder_selector_delay_epochs=2)
        self.assertIsInstance(model.feature_fusion[0], WNConv2d)
        self.assertFalse(model._selector_enabled)
        model.set_current_epoch(2)
        self.assertTrue(model._selector_enabled)
        with self.assertRaisesRegex(ValueError, 'Pruning learned fusion'):
            model.export_pruned_model([0])
        layer = model.feature_fusion[0]
        image = torch.randn(2, 80, 8, 8)
        with torch.no_grad():
            expected = layer(image)
            layer.weight.mul_(3)
            torch.testing.assert_close(layer(image), expected, atol=2e-6, rtol=2e-6)
        immediate = self.model(normalize_features=False, max_encoders=1)
        self.assertTrue(immediate._selector_enabled)
        immediate.eval()
        with torch.no_grad():
            immediate.encoder_selector.specialists_weights.copy_(torch.tensor([0., 1.]))
            selected = immediate(torch.randn(2, 3, 32, 32))
        for feature in selected:
            self.assertEqual(int(torch.count_nonzero(feature[:, :feature.shape[1] // 2])), 0)
            self.assertGreater(int(torch.count_nonzero(feature[:, feature.shape[1] // 2:])), 0)

    def test_specialist_state_is_loaded_strictly(self):
        bad_path = Path(self.temp.name) / 'bad.pth'
        state = dict(self.state)
        state.pop(next(iter(state)))
        torch.save([(3, state, {'backbone': 'convnextv2_atto', 'bands': ['R', 'G', 'B']})], bad_path)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            EosFM({'optical': ['R', 'G', 'B']}, 32, bad_path)


@unittest.skipUnless(os.environ.get('EOSFM_RELEASE_CHECKPOINT'), 'set EOSFM_RELEASE_CHECKPOINT for full artifact smoke')
class ReleaseInferenceTests(unittest.TestCase):
    def test_all_22_specialists_raw_parity_and_bn(self):
        torch.set_num_threads(4)
        model = EosFM({'optical': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
                                  'B9', 'B10', 'B11', 'B12', 'B8A']},
                      64, os.environ['EOSFM_RELEASE_CHECKPOINT'], normalize_features=False).eval()
        self.assertEqual(len(model.encoders), 22)
        self.assertIn('kenya-croptype', model.encoder_names)
        self.assertEqual(model.out_channels, [880, 1760, 3520, 7040])
        torch.manual_seed(625)
        image = torch.randn(2, 13, 64, 64)
        with torch.no_grad():
            raw = model(image)
            expected = [[] for _ in raw]
            for specialist, adapter in zip(model.encoders, model._smart_adapters):
                for level, tensor in enumerate(specialist(adapter.adapt(image))):
                    expected[level].append(tensor)
            for actual, tensors in zip(raw, expected):
                torch.testing.assert_close(actual, torch.cat(tensors, 1), rtol=0, atol=0)
                self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual([tuple(t.shape) for t in raw],
                         [(2, 880, 16, 16), (2, 1760, 8, 8), (2, 3520, 4, 4), (2, 7040, 2, 2)])
        model.normalize_features = True
        model.feature_normalizers = model._build_normalizers()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        bn = model.feature_normalizers[0][-1]
        before = bn.running_mean.clone()
        model.train()
        with torch.no_grad():
            model(image)
        self.assertTrue(all(not enc.training for enc in model.encoders))
        self.assertFalse(torch.equal(before, bn.running_mean))
        model.eval()
        before = bn.running_mean.clone()
        with torch.no_grad():
            model(image)
        self.assertTrue(torch.equal(before, bn.running_mean))
        del model
        gc.collect()


if __name__ == '__main__':
    unittest.main()
