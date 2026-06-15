"""Modelo de adyacencia compuesto por capacidades de entrenamiento e inferencia."""

from .dependencies import *  # noqa: F401,F403
from .persistence import PersistenceMixin
from .transitions import TransitionsMixin
from .simulation import SimulationMixin
from .occupancy import OccupancyMixin
from .graph import GraphMixin
from .training import TrainingMixin
from .supervised import SupervisedFilterMixin


class AIAdjacencyModel(PersistenceMixin, TransitionsMixin, SimulationMixin, OccupancyMixin, GraphMixin, TrainingMixin, SupervisedFilterMixin):
    def __init__(self) -> None:
        self.ready = False
        self.rooms: list[str] = []
        self.room_to_idx: dict[str, int] = {}
        self.transition_matrix = np.zeros((0, 0), dtype=np.float32)
        self.adjacency_neighbors: dict[str, list[str]] = {}
        self.adjacency_edges: list[dict[str, Any]] = []
        self.training_info: dict[str, Any] = {}
        self.sensor_room_map: dict[str, str] = {}
        self.transformer_model: Any | None = None
        self.transformer_device: Any | None = None
        self.transformer_context_length = TRANSFORMER_CONTEXT_LENGTH
        self.occupancy_transformer_model: Any | None = None
        self.occupancy_transformer_device: Any | None = None
        self.occupancy_transformer_rooms: list[str] = []
        self.occupancy_transformer_info: dict[str, Any] = {}
        self.occupancy_transformer_count_classes = 0
        self.real_profile_info: dict[str, Any] = {}
        self.pet_filter_model: Any | None = None
        self.pet_filter_device: Any | None = None
        self.pet_filter_info: dict[str, Any] = {}
        self.pet_filter_threshold = 0.0
        self.pet_filter_context_length = TRANSFORMER_CONTEXT_LENGTH
        self._lock = asyncio.Lock()
        self._train_lock = threading.Lock()

    def are_adjacent(self, a: str, b: str) -> bool:
        if not self.ready:
            return True
        if a == b:
            return True
        return b in self.adjacency_neighbors.get(a, [])

    def neighbors(self, room: str) -> list[str]:
        return self.adjacency_neighbors.get(room, [])
