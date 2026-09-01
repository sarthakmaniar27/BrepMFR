from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from unsupervised_training.checkpointing import extract_student_checkpoint
from unsupervised_training.config import ExperimentConfig
from unsupervised_training.constants import PACKAGE_ROOT


class CheckpointExtractionTest(unittest.TestCase):
    def test_only_student_deployable_weights_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint_path = root / "joint.ckpt"
            champion_path = root / "champion.ckpt"
            output_path = root / "student.ckpt"
            torch.save(
                {
                    "epoch": 2,
                    "global_step": 12,
                    "state_dict": {
                        "student.brep_encoder.weight": torch.tensor([1.0]),
                        "student.attention.weight": torch.tensor([2.0]),
                        "student.classifier.weight": torch.tensor([3.0]),
                        "student.class_weights": torch.ones(3),
                        "teacher.brep_encoder.weight": torch.tensor([9.0]),
                        "reconstruction_head.weight": torch.tensor([8.0]),
                    },
                },
                joint_path,
            )
            torch.save(
                {
                    "pytorch-lightning_version": "test",
                    "hyper_parameters": {"args": {"num_classes": 3}},
                    "state_dict": {},
                },
                champion_path,
            )
            extract_student_checkpoint(joint_path, champion_path, output_path)
            exported = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertEqual(
                set(exported["state_dict"]),
                {
                    "brep_encoder.weight",
                    "attention.weight",
                    "classifier.weight",
                    "class_weights",
                },
            )
            self.assertEqual(exported["epoch"], 2)
            self.assertEqual(exported["global_step"], 12)


class ConfigurationTest(unittest.TestCase):
    def test_shipped_configuration_is_valid(self):
        config = ExperimentConfig.from_json(
            PACKAGE_ROOT / "configs" / "abc_masked_geometry_v1.json"
        )
        self.assertEqual(config.num_classes, 3)
        self.assertGreater(config.masked_continuous_weight, 0.0)
        self.assertGreater(config.source_distillation_weight, 0.0)


if __name__ == "__main__":
    unittest.main()

