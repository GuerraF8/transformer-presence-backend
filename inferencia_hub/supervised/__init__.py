"""Preparación reproducible y entrenamiento supervisado de presencia."""

from .dataset import PreparedSupervisedDataset, SupervisedDatasetBuilder
from .manifest import TrainingManifestStore
from .trainer import SupervisedPresenceTrainer

__all__ = [
    "PreparedSupervisedDataset",
    "SupervisedDatasetBuilder",
    "SupervisedPresenceTrainer",
    "TrainingManifestStore",
]
