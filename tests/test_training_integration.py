"""Optional TerraTorch registration and configured-model smoke (no dataset/training).

EOSFM_TEST_TRAINING=1 python -m unittest discover -s tests -p test_training_integration.py
Requires the training extra and release/eosfm.pth.
"""
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('EOSFM_TEST_TRAINING'), 'set EOSFM_TEST_TRAINING with the training extra installed')
class TrainingIntegrationTests(unittest.TestCase):
    def test_registered_raw_classifier_configuration(self):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from eosfm import EosFM, register_terratorch
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, MODEL_FACTORY_REGISTRY

        register_terratorch()
        register_terratorch()
        self.assertIs(TERRATORCH_BACKBONE_REGISTRY['EosFM'], EosFM)
        self.assertIn('ChangeDetectionEncoderDecoderFactory', MODEL_FACTORY_REGISTRY)
        self.assertIn('TemporalEncoderDecoderFactory', MODEL_FACTORY_REGISTRY)
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
            cfg = compose(config_name='config', overrides=['experiment=test/test_eurosat'])
        # Keep this test independent of the shell working directory.
        cfg.model.model_args.backbone_encoder_weights = str(root / 'release/eosfm.pth')
        task = instantiate(cfg.model)
        encoder = next(module for module in task.modules() if isinstance(module, EosFM))
        self.assertEqual(len(encoder.encoders), 22)
        self.assertEqual(encoder.out_channels, [880, 1760, 3520, 7040])
        self.assertIsNone(encoder.feature_fusion_mode)
        self.assertFalse(encoder.normalize_features)
        self.assertFalse(encoder.projection_layer)
        self.assertIsNone(encoder.encoder_selector)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertEqual(str(cfg.trainer.precision), '32-true')


if __name__ == '__main__':
    unittest.main()
