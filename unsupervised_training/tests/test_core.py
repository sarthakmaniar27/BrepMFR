from __future__ import annotations

import unittest

import torch

from unsupervised_training.constants import IGNORE_LABEL
from unsupervised_training.graph_ops import (
    continuous_geometry_targets,
    masked_geometry_batch,
    sample_face_mask,
    soft_distillation_loss,
)
from unsupervised_training.prepare_unlabeled import validate_json_schema


class GraphOperationsTest(unittest.TestCase):
    def _batch(self):
        node_data = torch.zeros(5, 5, 5, 7)
        node_data[..., 0] = 0.25
        node_data[..., 3] = 1.0
        node_data[..., 6] = 1.0
        return {
            "padding_mask": torch.tensor([[False, False, True], [False, False, False]]),
            "node_data": node_data,
            "face_area": torch.arange(1, 6, dtype=torch.float32),
            "face_type": torch.tensor([1, 2, 3, 4, 5]),
            "face_loop": torch.tensor([1, 2, 3, 4, 5]),
            "in_degree": torch.tensor([2, 2, 3, 3, 4]),
            "out_degree": torch.tensor([2, 2, 3, 3, 4]),
            "label_feature": torch.full((5,), IGNORE_LABEL),
        }

    def test_mask_selects_at_least_one_face_per_graph(self):
        batch = self._batch()
        torch.manual_seed(7)
        mask = sample_face_mask(batch["padding_mask"], 0.15)
        self.assertEqual(mask.shape, (5,))
        self.assertGreaterEqual(int(mask[:2].sum()), 1)
        self.assertGreaterEqual(int(mask[2:].sum()), 1)

    def test_masking_does_not_modify_original(self):
        batch = self._batch()
        mask = torch.tensor([True, False, False, True, False])
        masked = masked_geometry_batch(batch, mask)
        self.assertEqual(float(batch["node_data"][0].sum()), 56.25)
        self.assertEqual(float(masked["node_data"][0].sum()), 0.0)
        self.assertEqual(int(masked["face_type"][0]), 0)
        self.assertEqual(int(batch["face_type"][0]), 1)

    def test_continuous_targets_shape_and_finiteness(self):
        targets = continuous_geometry_targets(self._batch())
        self.assertEqual(targets.shape, (5, 10))
        self.assertTrue(torch.isfinite(targets).all())

    def test_distillation_is_zero_for_identical_logits(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        loss = soft_distillation_loss(logits, logits, 2.0)
        self.assertLess(abs(float(loss)), 1.0e-6)


class JsonSchemaTest(unittest.TestCase):
    def test_label_is_not_required(self):
        face = {
            "id": 10,
            "uv": [0.0] * 175,
            "z": 1,
            "y": 0.5,
            "l": 1,
            "a": 0,
        }
        faces, edges = validate_json_schema({"faces": [face], "edges": []}, None)
        self.assertEqual((faces, edges), (1, 0))


if __name__ == "__main__":
    unittest.main()

