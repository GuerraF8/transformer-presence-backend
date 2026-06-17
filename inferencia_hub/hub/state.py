"""Inicialización del estado compartido por las operaciones de inferencia."""

from .dependencies import *  # noqa: F401,F403
from .layout import LayoutMixin
from .sensors import SensorsMixin
from .metrics import MetricsMixin
from .filtering import FilteringMixin
from .inference import InferenceMixin
from .events import EventsMixin
from .snapshot import SnapshotMixin
from .profiles import ProfilesMixin


class InferenceHubState(ProfilesMixin, LayoutMixin, SensorsMixin, MetricsMixin, FilteringMixin, InferenceMixin, EventsMixin, SnapshotMixin):
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.events: list[dict[str, Any]] = []
        self.rooms: set[str] = set()
        self.edge_support: Counter[tuple[str, str]] = Counter()
        self.last_active_by_room: dict[str, datetime] = {}
        self.occupancy_confirmed_by_room: dict[str, datetime] = {}
        self.active_sensor_types_by_room: dict[str, set[str]] = {}
        self.last_activation: LastActivation | None = None
        self.current_room: str | None = None
        self.current_active_rooms: list[str] = []
        self.latest_touched_edge: tuple[str, str] | None = None
        self.snapshot_publisher: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self.event_sink: Callable[
            [SensorEventInput, dict[str, Any], dict[str, Any]],
            Awaitable[None],
        ] | None = None
        self.rejected_transitions = 0

        self.reference_layout: dict[str, list[str]] = {}
        self.reference_layout_source = "auto"
        self.reference_layout_version = 0
        self.active_profile_id: str | None = None
        self.active_profile_name: str | None = None
        self.active_profile_revision: int | None = None
        self.active_profile_fingerprint: str | None = None
        self.active_profile_model_compatible = False
        self.active_profile_room_labels: dict[str, str] = {}

        self.non_adjacent_records: list[dict[str, Any]] = []
        self.max_non_adjacent_records = 2000
        self.non_adjacent_total = 0
        self.non_adjacent_multi_person = 0
        self.non_adjacent_pet_or_noise = 0
        self.non_adjacent_sensor_error = 0

        self.current_people_estimate = 0
        self.max_people_estimate = 0
        self.people_count_ground_truth: dict[str, Any] | None = None
        self.room_count_ground_truth: dict[str, dict[str, Any]] = {}
        self.ground_truth_samples: deque[dict[str, Any]] = deque(maxlen=5000)

        self.ingestion_latency_ms: deque[float] = deque(maxlen=5000)
        self.processing_latency_ms: deque[float] = deque(maxlen=5000)

        self.presence_hold_seconds = int(os.getenv("PRESENCE_HOLD_SECONDS", "180"))
        self.min_gap_seconds = int(os.getenv("MIN_GAP_SECONDS", "2"))
        self.max_gap_seconds = int(os.getenv("MAX_GAP_SECONDS", "600"))
        self.confirmed_edge_support = int(os.getenv("CONFIRMED_EDGE_SUPPORT", "2"))
        self.max_events_buffer = int(os.getenv("MAX_EVENTS_BUFFER", "30000"))
        self.include_all_state_transitions = os.getenv("INCLUDE_ALL_STATE_TRANSITIONS", "1") != "0"
        self.presence_filter_enabled = os.getenv("PET_FILTER_ENABLED", "1") != "0"
        self.presence_filter_window_seconds = max(1, min(600, int(os.getenv("PET_FILTER_WINDOW_SECONDS", "20"))))
        self.presence_filter_min_motion_events = max(1, min(20, int(os.getenv("PET_FILTER_MIN_EVENTS", "2"))))
        self.presence_filter_min_distinct_rooms = max(1, min(20, int(os.getenv("PET_FILTER_MIN_DISTINCT_ROOMS", "1"))))
        self.presence_filter_events: deque[dict[str, Any]] = deque(maxlen=512)
        self.presence_filter_suppressed_total = 0

        self.input_mode = "listen"
        self.replay_task: asyncio.Task | None = None
        self.replay_paused = False
        self.replay_stop_requested = False
        self.replay_step_budget = 0
        self.replay_total_events = 0
        self.replay_processed_events = 0
        self.replay_last_error: str | None = None
        self.last_replay_config: dict[str, Any] = {}

        self.ai_model = AIAdjacencyModel()
        self.presence_belief = np.zeros((0,), dtype=np.float32)
        self.sequence_history: deque[EventRecord] = deque(maxlen=512)
        self.real_sensor_rooms: set[str] = set()
        self.real_sensor_assignments: dict[str, dict[str, Any]] = {}
        self.real_sensor_require_explicit_selection = True
        self.real_sensor_rejected_events = 0
        self.real_sensor_last_rejected: dict[str, Any] | None = None
