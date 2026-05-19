import torch
import numpy as np
import json
from planner.base import BasePlanner
from rl_planner.dqn_agent import QNetwork


class DQNPlanner(BasePlanner):
    """Планировщик, использующий обученную DQN-сеть для выбора вех."""

    def __init__(self, model_path: str, flags_order_path: str, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_net.eval()
        self.action_dim = action_dim
        self.beacon_to_idx = None
        self.idx_to_beacon = None
        with open(flags_order_path, "r") as f:
            self.flags_order = json.load(f)

    def set_mapping(self, beacons: list[dict]):
        self.beacon_to_idx = {b["id"]: i for i, b in enumerate(beacons)}
        self.idx_to_beacon = {i: b["id"] for i, b in enumerate(beacons)}

    def _state_to_vector(self, state):
        vec = []
        for k in self.flags_order:
            vec.append(1.0 if state.has_flag(k) else 0.0)
        return np.array(vec, dtype=np.float32)

    def select_beacon(self, state: "GameState", available_beacons: list[dict]) -> str:
        if not self.beacon_to_idx:
            # fallback
            return available_beacons[0]["id"]
        valid_ids = [b["id"] for b in available_beacons]
        valid_actions = [self.beacon_to_idx[i] for i in valid_ids]
        state_vec = self._state_to_vector(state)
        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze().cpu().numpy()
        mask = np.full(self.action_dim, -np.inf)
        for a in valid_actions:
            mask[a] = 0
        q_values = q_values + mask
        best_action_idx = int(np.argmax(q_values))
        return self.idx_to_beacon[best_action_idx]
