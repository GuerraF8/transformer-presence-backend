from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.ai import dependencies as ai_dependencies
from inferencia_hub.runtime import lifecycle
from inferencia_hub.supervised.artifact import (
    packaged_pet_filter_dir,
)


class SupervisedModelLifecycleTest(unittest.TestCase):
    def test_packaged_filter_metadata_matches_checkpoint(
        self,
    ) -> None:
        directory = packaged_pet_filter_dir()
        metadata = json.loads(
            (directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = directory / "pet_motion_transformer.pt"
        self.assertTrue(checkpoint.exists())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            metadata["checkpoint_sha256"],
        )
        self.assertTrue(
            metadata["pet_filter_info"][
                "suppression_enabled"
            ]
        )
        self.assertEqual(
            metadata["pet_filter_info"]["source"],
            "bundled",
        )

    @unittest.skipUnless(
        ai_dependencies.TORCH_AVAILABLE,
        "PyTorch no está instalado",
    )
    def test_packaged_filter_loads_without_training(
        self,
    ) -> None:
        model = AIAdjacencyModel()
        loaded = model.load_packaged_pet_filter()
        self.assertTrue(loaded["loaded"])
        self.assertIsNotNone(model.pet_filter_model)
        self.assertTrue(
            model.pet_filter_info["suppression_enabled"]
        )
        self.assertEqual(
            model.pet_filter_info["source"],
            "bundled",
        )

    def test_previous_recall_guard_is_migrated_to_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "pet_filter_info": {
                            "enabled": True,
                            "suppression_enabled": False,
                            "test": {
                                "activation_guard": (
                                    "recall humano inferior"
                                )
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            restored = AIAdjacencyModel()
            restored.load_state(path)
            self.assertTrue(
                restored.pet_filter_info[
                    "suppression_enabled"
                ]
            )
            self.assertEqual(
                restored.pet_filter_info[
                    "activation_policy"
                ],
                "operational_preference",
            )

    def test_atomic_activation_and_bidirectional_rollback(self) -> None:
        original_model = lifecycle.hub_state.ai_model
        original_profile_id = lifecycle.hub_state.active_profile_id
        original_fingerprint = (
            lifecycle.hub_state.active_profile_fingerprint
        )
        original_compatible = (
            lifecycle.hub_state.active_profile_model_compatible
        )
        try:
            lifecycle.hub_state.ai_model = AIAdjacencyModel()
            lifecycle.hub_state.active_profile_id = "profile-test"
            lifecycle.hub_state.active_profile_fingerprint = "fingerprint"
            with tempfile.TemporaryDirectory() as directory:
                with patch.dict(
                    os.environ,
                    {"MODEL_STATE_DIR": directory},
                    clear=False,
                ):
                    lifecycle.hub_state.ai_model.training_info = {
                        "supervised": {"run_id": "old"}
                    }
                    first = lifecycle.persist_model_state()
                    self.assertIsNotNone(first)

                    lifecycle.hub_state.ai_model.training_info = {
                        "supervised": {"run_id": "new"}
                    }
                    activated = lifecycle.persist_model_state_atomic()
                    self.assertIsNotNone(activated)
                    self.assertTrue(
                        Path(activated["previous_dir"]).exists()
                    )

                    lifecycle.rollback_model_state()
                    self.assertEqual(
                        lifecycle.hub_state.ai_model.training_info[
                            "supervised"
                        ]["run_id"],
                        "old",
                    )
                    lifecycle.rollback_model_state()
                    self.assertEqual(
                        lifecycle.hub_state.ai_model.training_info[
                            "supervised"
                        ]["run_id"],
                        "new",
                    )
        finally:
            lifecycle.hub_state.ai_model = original_model
            lifecycle.hub_state.active_profile_id = original_profile_id
            lifecycle.hub_state.active_profile_fingerprint = (
                original_fingerprint
            )
            lifecycle.hub_state.active_profile_model_compatible = (
                original_compatible
            )


if __name__ == "__main__":
    unittest.main()
