# precision_climate_robot.py
"""
Precision Climate Robot — v1.0.0
Multi-sensor phase-coherence brain for benign climate robotics
(tree planting, reef monitoring, glacier sensing, etc.)

MIT License — Agape Intelligence, November 26 2025
"""

from __future__ import annotations
import math
import os
from typing import Optional, Sequence, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# ———————————————————————— Config ————————————————————————
DEFAULT_SR = 200
DEFAULT_N_BANDS = 64
MOOD_DIM = 128
DEVICE = "cpu"


# ———————————————————————— Utilities ————————————————————————
def log_freq_bands(n_bands: int = DEFAULT_N_BANDS, f_min: float = 0.5, f_max: float = 100.0, sample_rate: int = DEFAULT_SR) -> List[Tuple[float, float]]:
    nyquist = sample_rate / 2.0
    f_max = min(f_max, nyquist * 0.95)
    boundaries = torch.logspace(math.log10(f_min), math.log10(f_max), n_bands + 1, base=10.0)
    return [(float(boundaries[i].item()), float(boundaries[i + 1].item())) for i in range(n_bands)]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


# ———————————————————————— Phase extraction ————————————————————————
def analytic_signal_hilbert(x: Tensor) -> Tensor:
    X = torch.fft.rfft(x, dim=-1)
    n = x.shape[-1]
    h = torch.ones_like(X, dtype=torch.float32)
    if X.shape[-1] > 2:
        h[..., 1:-1] = 2.0
    analytic = torch.fft.irfft(X * h, n=n, dim=-1)
    return torch.complex(analytic, torch.zeros_like(analytic))


def instantaneous_phase_bandwise(signal: Tensor, sample_rate: int = DEFAULT_SR, bands: Optional[Sequence[Tuple[float,float]]] = None) -> Tensor:
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    signal = signal - signal.mean(dim=-1, keepdim=True)
    signal = signal * torch.hann_window(signal.shape[-1], device=signal.device)

    if bands is None:
        return torch.angle(analytic_signal_hilbert(signal)).unsqueeze(-2)

    X_full = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(signal.shape[-1], d=1.0/sample_rate, device=signal.device)
    phases = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        Xb = X_full.clone()
        Xb[..., ~mask] = 0
        band_td = torch.fft.irfft(Xb, n=signal.shape[-1], dim=-1)
        analytic = analytic_signal_hilbert(band_td)
        phases.append(torch.angle(analytic))
    return torch.stack(phases, dim=-2)


def multimodal_phase_stack(sensors: Dict[str, np.ndarray], sr_map: Dict[str, int], bands_map: Optional[Dict[str, Sequence[Tuple[float,float]]]] = None, device: str = DEVICE) -> Tensor:
    device_t = torch.device(device)
    all_phases = []
    for name, arr in sensors.items():
        sr = sr_map.get(name, DEFAULT_SR)
        bands = bands_map.get(name) if bands_map else None
        t = torch.from_numpy(arr).float().to(device_t)
        phases = instantaneous_phase_bandwise(t, sample_rate=sr, bands=bands)
        last = phases[..., -1].squeeze(0)
        all_phases.append(last)
    return torch.cat(all_phases, dim=0)


# ———————————————————————— Adaptive hysteresis ————————————————————————
class AdaptiveHysteresis:
    def __init__(self, n_channels: int):
        self.prev_phase = torch.zeros(n_channels)
        self.prev_alpha = torch.full((n_channels,), 0.22)

    def smooth(self, phase_curr: Tensor) -> Tensor:
        diff = (phase_curr - self.prev_phase + math.pi) % (2*math.pi) - math.pi
        vel = diff.abs()
        target_alpha = 0.08 + 0.37 * torch.sigmoid(8.0 * vel - 1.5)
        alpha = self.prev_alpha + 0.12 * (target_alpha - self.prev_alpha)
        smoothed = alpha * phase_curr + (1.0 - alpha) * self.prev_phase
        self.prev_phase = smoothed.clone()
        self.prev_alpha = alpha.clone()
        return smoothed


# ———————————————————————— MoodVector128 ————————————————————————
class MoodVector128(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels + 1, 256), nn.SiLU(),
            nn.Linear(256, 192), nn.SiLU(),
            nn.Linear(192, MOOD_DIM), nn.Tanh()
        )

    def forward(self, phases: Tensor, coherence: Tensor) -> Tensor:
        x = torch.cat([phases / math.pi, coherence.unsqueeze(-1)], dim=-1)
        return self.net(x)


# ———————————————————————— Policy ————————————————————————
class AffordancePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MOOD_DIM, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 8), nn.Tanh()
        )

    def forward(self, mv: Tensor) -> Tensor:
        self.net(mv)


# ———————————————————————— Main Brain ————————————————————————
class PrecisionClimateBrain:
    def __init__(self, sensors_spec: Dict[str, int], bands_map: Optional[Dict[str, Sequence[Tuple[float,float]]]] = None, device: str = DEVICE):
        self.device = device
        self.sr_map = sensors_spec
        self.bands_map = bands_map or {}
        idx = 0
        self.channel_slices = {}
        for name, sr in sensors_spec.items():
            bands = self.bands_map.get(name, log_freq_bands(DEFAULT_N_BANDS, sample_rate=sr))
            n_b = len(bands)
            self.channel_slices[name] = (idx, n_b)
            idx += n_b
        self.n_channels = idx

        self.hyst = AdaptiveHysteresis(self.n_channels)
        self.mood_model = MoodVector128(self.n_channels)
        self.policy = AffordancePolicy()
        self.leader_score = 0.0

    def compute(self, sensors_window: Dict[str, np.ndarray]) -> Dict:
        phases = multimodal_phase_stack(sensors_window, self.sr_map, self.bands_map, self.device)
        if phases.shape[0] != self.n_channels:
            pad = torch.zeros(self.n_channels - phases.shape[0], device=phases.device)
            phases = torch.cat([phases, pad]) if phases.shape[0] < self.n_channels else phases[:self.n_channels]

        phases_smoothed = self.hyst.smooth(phases)

        p = phases_smoothed
        p1, p2, p3 = p[0::3], p[1::3], p[2::3]
        minlen = min(p1.shape[0], p2.shape[0], p3.shape[0])
        triad = torch.exp(1j * (p1[:minlen] - p2[:minlen] + p3[:minlen]))
        coherence = float(torch.abs(torch.mean(triad)).clamp(0.0, 1.0))

        mv = self.mood_model(phases_smoothed, torch.tensor(coherence))
        action_raw = self.policy(mv)

        vx = clamp(action_raw[0].item() * 2.0, -3.0, 3.0)
        vy = clamp(action_raw[1].item() * 2.0, -3.0, 3.0)
        dz = clamp(action_raw[2].item() * 1.0, -2.0, 2.0)
        precision_score = clamp((action_raw[3].item() + 1.0) / 2.0, 0.0, 1.0)
        hold_conf = clamp((action_raw[4].item() + 1.0) / 2.0, 0.0, 1.0)
        broadcast = clamp((action_raw[5].item() + 1.0) / 2.0, 0.0, 1.0)

        self.leader_score = max(self.leader_score, coherence * precision_score)

        return {
            "mood_vector": mv.detach().cpu().numpy(),
            "coherence": coherence,
            "action": {"vx": vx, "vy": vy, "dz": dz, "precision_score": precision_score,
                       "hold_conf": hold_conf, "broadcast": broadcast},
            "leader_score": self.leader_score
        }

    def export_action_head(self, path: str = "exports/precision_action_head.pt"):
        class Wrapper(nn.Module):
            def __init__(self, mood, policy):
                super().__init__()
                self.mood = mood
                self.policy = policy
            def forward(self, phases: Tensor, coherence: Tensor) -> Tensor:
                mv = self.mood(phases, coherence)
                return self.policy(mv)

        wrapper = Wrapper(self.mood_model, self.policy).eval()
        scripted = torch.jit.script(wrapper)
        os.makedirs(os.path.dirname(path) if "/" in path else ".", exist_ok=True)
        scripted.save(path)
        print(f"TorchScript model exported → {path}")


# ———————————————————————— Demo + Auto-export ————————————————————————
def _demo():
    T = DEFAULT_SR
    t = np.linspace(0, 1.0, T, endpoint=False)
    mic = 0.6*np.sin(2*np.pi*6*t) + 0.2*np.sin(2*np.pi*14*t) + 0.05*np.random.randn(T)
    imu = 0.1*np.sin(2*np.pi*12*t) + 0.02*np.random.randn(T)
    depth = np.convolve(np.random.randn(T), np.ones(20)/20, mode='same') * 0.05

    sensors_spec = {'mic': DEFAULT_SR, 'imu': DEFAULT_SR, 'depth': 10}
    bands_map = {
        'mic': log_freq_bands(64, f_max=80.0, sample_rate=DEFAULT_SR),
        'imu': log_freq_bands(32, f_max=60.0, sample_rate=DEFAULT_SR),
        'depth': log_freq_bands(8, f_min=0.01, f_max=2.0, sample_rate=10),
    }

    brain = PrecisionClimateBrain(sensors_spec, bands_map)
    sensors = {'mic': mic.astype(np.float32), 'imu': imu.astype(np.float32), 'depth': depth.astype(np.float32)}
    result = brain.compute(sensors)
    print("Action:", result['action'])
    print("Coherence:", f"{result['coherence']:.4f}")
    print("Leader score:", f"{result['leader_score']:.4f}")


if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    print("PrecisionClimateRobot v1.0.0 — November 26 2025")
    print("Running demo and exporting TorchScript model...\n")
    
    _demo()

    sensors_spec = {'mic': DEFAULT_SR, 'imu': DEFAULT_SR, 'depth': 10}
    bands_map = {
        'mic': log_freq_bands(64, f_max=80.0, sample_rate=DEFAULT_SR),
        'imu': log_freq_bands(32, f_max=60.0, sample_rate=DEFAULT_SR),
        'depth': log_freq_bands(8, f_min=0.01, f_max=2.0, sample_rate=10),
    }

    brain = PrecisionClimateBrain(sensors_spec, bands_map, device="cpu")
    export_path = "exports/precision_action_head.pt"
    brain.export_action_head(export_path)

    size_kb = os.path.getsize(export_path) / 1024
    print(f"\nDeployment ready:")
    print(f"→ {export_path} ({size_kb:.1f} KB)")
    print("Model is now portable to Android, iOS, Raspberry Pi, Jetson, or any TorchScript runtime.")
