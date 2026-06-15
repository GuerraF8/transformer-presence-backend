"""Persistencia y recuperación del estado de los modelos."""

from .dependencies import *  # noqa: F401,F403


class PersistenceMixin:
    def save_state(self, model_dir: str | Path) -> dict[str, Any]:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        core_path = path / "model_state.json"
        transition_path = path / "transition_matrix.npy"
        transformer_path = path / "next_room_transformer.pt"
        occupancy_path = path / "occupancy_transformer.pt"
        pet_filter_path = path / "pet_motion_transformer.pt"

        payload = {
            "schema_version": 2,
            "ready": self.ready,
            "rooms": self.rooms,
            "adjacency_neighbors": self.adjacency_neighbors,
            "adjacency_edges": self.adjacency_edges,
            "training_info": self.training_info,
            "sensor_room_map": self.sensor_room_map,
            "transformer_context_length": self.transformer_context_length,
            "occupancy_transformer_rooms": self.occupancy_transformer_rooms,
            "occupancy_transformer_info": self.occupancy_transformer_info,
            "occupancy_transformer_count_classes": self.occupancy_transformer_count_classes,
            "real_profile_info": self.real_profile_info,
            "pet_filter_info": self.pet_filter_info,
            "pet_filter_threshold": self.pet_filter_threshold,
            "pet_filter_context_length": self.pet_filter_context_length,
        }
        core_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(transition_path, self.transition_matrix)

        saved = {
            "core_path": str(core_path),
            "transition_path": str(transition_path),
            "transformer_path": None,
            "occupancy_transformer_path": None,
            "pet_filter_path": None,
        }
        if HF_AVAILABLE and self.transformer_model is not None:
            torch.save({"state_dict": self.transformer_model.state_dict()}, transformer_path)
            saved["transformer_path"] = str(transformer_path)
        if HF_AVAILABLE and self.occupancy_transformer_model is not None:
            torch.save({"state_dict": self.occupancy_transformer_model.state_dict()}, occupancy_path)
            saved["occupancy_transformer_path"] = str(occupancy_path)
        if TORCH_AVAILABLE and self.pet_filter_model is not None:
            torch.save(
                {"state_dict": self.pet_filter_model.state_dict()},
                pet_filter_path,
            )
            saved["pet_filter_path"] = str(pet_filter_path)
        return saved

    def load_state(self, model_dir: str | Path) -> dict[str, Any]:
        path = Path(model_dir)
        core_path = path / "model_state.json"
        transition_path = path / "transition_matrix.npy"
        transformer_path = path / "next_room_transformer.pt"
        occupancy_path = path / "occupancy_transformer.pt"
        pet_filter_path = path / "pet_motion_transformer.pt"
        if not core_path.exists():
            return {"loaded": False, "reason": "sin estado persistido"}

        payload = json.loads(core_path.read_text(encoding="utf-8"))
        self.ready = bool(payload.get("ready"))
        self.rooms = [str(room) for room in payload.get("rooms", [])]
        self.room_to_idx = {room: idx for idx, room in enumerate(self.rooms)}
        self.adjacency_neighbors = {
            str(room): [str(nb) for nb in neighbors]
            for room, neighbors in dict(payload.get("adjacency_neighbors", {})).items()
        }
        self.adjacency_edges = list(payload.get("adjacency_edges", []))
        self.training_info = dict(payload.get("training_info", {}))
        self.sensor_room_map = {
            str(entity_id): str(room)
            for entity_id, room in dict(payload.get("sensor_room_map", {})).items()
        }
        self.transformer_context_length = int(payload.get("transformer_context_length") or TRANSFORMER_CONTEXT_LENGTH)
        self.occupancy_transformer_rooms = [
            str(room) for room in payload.get("occupancy_transformer_rooms", [])
        ]
        self.occupancy_transformer_info = dict(payload.get("occupancy_transformer_info", {}))
        self.occupancy_transformer_count_classes = int(payload.get("occupancy_transformer_count_classes") or 0)
        self.real_profile_info = dict(payload.get("real_profile_info", {}))
        self.pet_filter_info = dict(payload.get("pet_filter_info", {}))
        if (
            self.pet_filter_info.get("enabled")
            and not self.pet_filter_info.get("suppression_enabled")
            and dict(self.pet_filter_info.get("test") or {}).get(
                "activation_guard"
            )
        ):
            self.pet_filter_info["suppression_enabled"] = True
            self.pet_filter_info["activation_policy"] = (
                "operational_preference"
            )
            self.pet_filter_info["previous_activation_guard"] = (
                self.pet_filter_info["test"].pop(
                    "activation_guard"
                )
            )
        self.pet_filter_threshold = float(payload.get("pet_filter_threshold") or 0.0)
        self.pet_filter_context_length = int(
            payload.get("pet_filter_context_length") or TRANSFORMER_CONTEXT_LENGTH
        )

        if transition_path.exists():
            self.transition_matrix = np.load(transition_path).astype(np.float32)
        elif self.rooms:
            self.transition_matrix = np.eye(len(self.rooms), dtype=np.float32)
        else:
            self.transition_matrix = np.zeros((0, 0), dtype=np.float32)

        loaded_models: list[str] = []
        if HF_AVAILABLE and transformer_path.exists() and self.rooms:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NextRoomTransformer(input_size=len(self.rooms) + 11, room_count=len(self.rooms)).to(device)
            checkpoint = torch.load(transformer_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.transformer_model = model
            self.transformer_device = device
            loaded_models.append("next_room_transformer")

        if HF_AVAILABLE and occupancy_path.exists() and self.occupancy_transformer_rooms:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            count_classes = max(1, self.occupancy_transformer_count_classes)
            rooms = self.occupancy_transformer_rooms
            model = OccupancyTransformer(
                input_size=len(rooms) + 11,
                room_count=len(rooms),
                count_classes=count_classes,
            ).to(device)
            checkpoint = torch.load(occupancy_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.occupancy_transformer_model = model
            self.occupancy_transformer_device = device
            loaded_models.append("occupancy_transformer")

        if (
            TORCH_AVAILABLE
            and pet_filter_path.exists()
            and self.pet_filter_info.get("enabled")
        ):
            from ..models.pet_filter import PetMotionTransformer

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PetMotionTransformer(
                input_size=int(self.pet_filter_info.get("input_size") or 12),
                context_length=self.pet_filter_context_length,
            ).to(device)
            checkpoint = torch.load(pet_filter_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.pet_filter_model = model
            self.pet_filter_device = device
            loaded_models.append("pet_motion_transformer")

        return {
            "loaded": True,
            "rooms": len(self.rooms),
            "models": loaded_models,
            "core_path": str(core_path),
        }
