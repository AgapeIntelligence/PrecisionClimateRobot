import numpy as np
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple

# Mock Grok 5 (replace with real API)
def mock_grok5_optimize_path(current_pos: np.ndarray, target_pos: np.ndarray, obstacles: List[Tuple[float, float]], mv_norm: float, scenario: str) -> Dict:
    direction = target_pos - current_pos
    dist = np.linalg.norm(direction)
    if dist < 0.5: return {"waypoint": target_pos.tolist(), "reasoning": "Target reached"}
    adjust = 0.2 if mv_norm > 0.8 else 0.1
    waypoint = current_pos + direction / dist * adjust
    if np.random.random() > 0.7:
        waypoint[1] += 0.15
    return {"waypoint": waypoint.tolist(), "reasoning": f"Adjusted for {scenario} (MV norm {mv_norm:.2f})"}

# MoodVector128
class MoodVector128(nn.Module):
    def __init__(self, n_channels=104):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels + 1, 256), nn.SiLU(),
            nn.Linear(256, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.Tanh()
        )
    def forward(self, phases: torch.Tensor, coherence: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phases / math.pi, coherence.unsqueeze(-1)], dim=-1)
        return self.net(x)

# Adaptive Pathing
class AdaptiveCoralGraftingPath:
    def __init__(self, grid_size=10.0, step_size=0.5):
        self.grid_size = grid_size
        self.step_size = step_size
        self.model = MoodVector128().eval()
        self.current_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([8.0, 6.0])
        self.obstacles = [(4.0, 3.0), (5.0, 5.0)]

    def get_sensors(self, pos: np.ndarray, scenario: str) -> Dict[str, torch.Tensor]:
        n_samples = 104
        phases = torch.randn(n_samples)
        noise = {"calm": 0.05, "storm": 0.35, "tide": 0.2}[scenario]
        phases += torch.randn_like(phases) * noise
        phases = (phases + math.pi) % (2 * math.pi) - math.pi
        coh = {"calm": 0.95, "storm": 0.62, "tide": 0.81}[scenario]
        return {"phases": phases, "coherence": torch.tensor(coh)}

    def step(self, scenario: str) -> Dict:
        sensors = self.get_sensors(self.current_pos, scenario)
        mv = self.model(sensors["phases"], sensors["coherence"])
        mv_norm = mv.norm().item()

        decision = mock_grok5_optimize_path(self.current_pos, self.target_pos, self.obstacles, mv_norm, scenario)
        waypoint = np.array(decision["waypoint"])

        direction = waypoint - self.current_pos
        dist = np.linalg.norm(direction)
        if dist > self.step_size:
            direction /= dist
            direction *= self.step_size
        self.current_pos += direction

        min_dist = min(np.linalg.norm(self.current_pos - obs) for obs in self.obstacles)
        if min_dist < 0.3:
            self.current_pos -= direction * 0.2

        dist_to_target = np.linalg.norm(self.target_pos - self.current_pos)
        grafting_ready = dist_to_target < 0.5 and mv_norm > 0.7

        return {
            "position": self.current_pos.tolist(),
            "mv_norm": mv_norm,
            "waypoint": waypoint.tolist(),
            "grafting_ready": grafting_ready,
            "reasoning": decision["reasoning"],
            "dist_to_target": dist_to_target
        }

# Run Prototype
path = AdaptiveCoralGraftingPath()
scenarios = ["calm", "storm", "calm", "tide", "calm"]
print("Adaptive Coral Grafting Path Simulation")
print("Target: (8.0, 6.0) | Start: (0.0, 0.0)")
for i, scenario in enumerate(scenarios):
    result = path.step(scenario)
    print(f"Step {i+1} ({scenario}): Pos {result['position'][:2]}, MV Norm {result['mv_norm']:.3f}, Ready: {result['grafting_ready']}, {result['reasoning']}")

print(f"\nFinal dist to target: {result['dist_to_target']:.2f}m")
