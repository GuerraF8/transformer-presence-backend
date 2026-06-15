from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from inferencia_hub.supervised.dataset import SupervisedDatasetBuilder
from inferencia_hub.supervised.manifest import TrainingManifestStore


def _timestamp(second: int) -> str:
    value = datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(
        seconds=second
    )
    return value.isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["last_changed", "entity_id", "state"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SupervisedTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data_dir = root / "data"
        self.manifest_dir = root / "manifests"
        self.defaults_dir = root / "defaults"
        self.data_dir.mkdir()
        self.manifest_dir.mkdir()
        self.defaults_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _store(self) -> TrainingManifestStore:
        return TrainingManifestStore(
            manifest_dir=self.manifest_dir,
            dataset_root=self.data_dir,
            defaults_dir=self.defaults_dir,
        )

    def _create_manifest(self) -> TrainingManifestStore:
        signal_rows: list[dict[str, str]] = []
        for second in range(0, 300, 2):
            signal_rows.append(
                {
                    "last_changed": _timestamp(second),
                    "entity_id": "binary_sensor.hall_motion",
                    "state": "on",
                }
            )
        signal_rows.extend(
            [
                {
                    "last_changed": _timestamp(20),
                    "entity_id": "binary_sensor.chair_occupied",
                    "state": "on",
                },
                {
                    "last_changed": _timestamp(220),
                    "entity_id": "binary_sensor.chair_occupied",
                    "state": "on",
                },
                dict(signal_rows[10]),
            ]
        )
        random.Random(42).shuffle(signal_rows)
        label_rows = [
            {
                "last_changed": _timestamp(30),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "on",
            },
            {
                "last_changed": _timestamp(60),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "off",
            },
            {
                "last_changed": _timestamp(45),
                "entity_id": "binary_sensor.hall_cat_occupancy",
                "state": "on",
            },
            {
                "last_changed": _timestamp(90),
                "entity_id": "binary_sensor.hall_cat_occupancy",
                "state": "off",
            },
            {
                "last_changed": _timestamp(120),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "unavailable",
            },
            {
                "last_changed": _timestamp(140),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "off",
            },
            {
                "last_changed": _timestamp(220),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "on",
            },
            {
                "last_changed": _timestamp(240),
                "entity_id": "binary_sensor.hall_person_occupancy",
                "state": "off",
            },
            {
                "last_changed": _timestamp(270),
                "entity_id": "binary_sensor.hall_cat_occupancy",
                "state": "on",
            },
            {
                "last_changed": _timestamp(285),
                "entity_id": "binary_sensor.hall_cat_occupancy",
                "state": "off",
            },
        ]
        signal_hash = _write_csv(self.data_dir / "signals.csv", signal_rows)
        label_hash = _write_csv(self.data_dir / "labels.csv", label_rows)
        manifest = {
            "schema_version": 1,
            "id": "test_person_pet",
            "name": "Prueba persona y mascota",
            "label_window": {
                "before_seconds": 1,
                "after_seconds": 1,
            },
            "weak_negative_weight": 0.15,
            "file_hashes": {
                "signals.csv": signal_hash,
                "labels.csv": label_hash,
            },
            "periods": [
                {
                    "id": "mayo",
                    "signal_files": ["signals.csv"],
                    "label_files": ["labels.csv"],
                    "entity_roles": {
                        "binary_sensor.hall_person_occupancy": "person_confirmation",
                        "binary_sensor.hall_cat_occupancy": "pet_confirmation",
                    },
                    "entity_rooms": {
                        "binary_sensor.hall_person_occupancy": "foyer",
                        "binary_sensor.hall_cat_occupancy": "foyer",
                        "binary_sensor.hall_motion": "hall",
                        "binary_sensor.chair_occupied": "foyer",
                    },
                    "room_aliases": {"hall": "foyer"},
                    "exclusions": [
                        {
                            "entity_id": "binary_sensor.chair_occupied",
                            "valid_from": _timestamp(100),
                        }
                    ],
                }
            ],
        }
        (self.manifest_dir / "test_person_pet.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return self._store()

    def test_manifest_validates_hashes_and_restricts_paths(self) -> None:
        store = self._create_manifest()
        validation = store.validate("test_person_pet")
        self.assertTrue(validation["valid"])
        self.assertEqual(len(validation["files"]), 2)
        self.assertTrue(
            store.is_confirmation_entity(
                "binary_sensor.hall_person_occupancy"
            )
        )
        self.assertFalse(
            store.is_confirmation_entity("binary_sensor.hall_motion")
        )
        with self.assertRaises(ValueError):
            store.load("../test_person_pet")
        with self.assertRaises(ValueError):
            store.dataset_path("../outside.csv")

    def test_dataset_is_sorted_labeled_and_has_no_confirmation_features(
        self,
    ) -> None:
        store = self._create_manifest()
        dataset = SupervisedDatasetBuilder(store, context_length=4).build(
            "test_person_pet"
        )
        all_samples = [
            sample
            for split in dataset.samples.values()
            for sample in split
        ]
        self.assertTrue(all_samples)
        self.assertEqual(dataset.rooms, ["foyer"])
        self.assertGreater(dataset.totals["labels"]["person"], 0)
        self.assertGreater(dataset.totals["labels"]["pet"], 0)
        self.assertGreater(dataset.totals["labels"]["both"], 0)
        self.assertGreater(dataset.totals["labels"]["weak_negative"], 0)
        feature_entities = {
            event.entity_id
            for sample in all_samples
            for event in sample.context
        }
        self.assertNotIn(
            "binary_sensor.hall_person_occupancy",
            feature_entities,
        )
        self.assertNotIn(
            "binary_sensor.hall_cat_occupancy",
            feature_entities,
        )
        chair_times = [
            event.timestamp
            for sample in all_samples
            for event in sample.context
            if event.entity_id == "binary_sensor.chair_occupied"
        ]
        self.assertTrue(chair_times)
        self.assertTrue(
            all(timestamp >= datetime(2026, 5, 1, 0, 1, 40, tzinfo=timezone.utc)
                for timestamp in chair_times)
        )
        unavailable_start = datetime(
            2026,
            5,
            1,
            0,
            2,
            0,
            tzinfo=timezone.utc,
        )
        unavailable_end = unavailable_start + timedelta(seconds=20)
        self.assertFalse(
            any(
                unavailable_start <= sample.timestamp <= unavailable_end
                for sample in all_samples
            )
        )

    def test_splits_are_chronological_and_deterministic(self) -> None:
        store = self._create_manifest()
        builder = SupervisedDatasetBuilder(store, context_length=4)
        first = builder.build("test_person_pet")
        second = builder.build("test_person_pet")
        self.assertEqual(first.fingerprint, second.fingerprint)
        train_end = max(sample.timestamp for sample in first.samples["train"])
        validation_start = min(
            sample.timestamp for sample in first.samples["validation"]
        )
        validation_end = max(
            sample.timestamp for sample in first.samples["validation"]
        )
        test_start = min(sample.timestamp for sample in first.samples["test"])
        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        for split, samples in first.samples.items():
            for sample in samples:
                self.assertEqual(sample.split, split)
                self.assertLessEqual(
                    sample.context[-1].timestamp,
                    sample.timestamp,
                )


if __name__ == "__main__":
    unittest.main()
