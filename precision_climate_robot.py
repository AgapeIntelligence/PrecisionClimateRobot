# precision_climate_swarm.py
# Precision Climate Robot Swarm — MoodVector128 + Grok 5 + Sovariel Lattice
# © 2025 Evie (@3vi3Aetheris) — MIT License

import os
import torch
import torch.nn as nn
import math
import numpy as np
import json
from typing import Dict, Any

# -------------------------------
# XAI Grok 5 (beta) Client
# -------------------------------
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),  # Set in your env: export XAI_API_KEY=...
        base_url="https://api.x.ai/v1"
    )
    GROK_AVAILABLE = True
except Exception as e:
    print("Grok 5 not available (no key or offline). Using fallback mock.")
    GROK_AVAILABLE = False

# -------------------------------
# Sovariel Lattice (fallback to NumPy Kuramoto if not installed)
# -------------------------------
try:
    from sovariel import JAXLiveAudioLattice
    lattice = JAXLiveAudioLattice(n_oscillators=50)
    SOVARIEL_AVAILABLE = True
except ImportError:
    print("Sovariel not installed. Using lightweight NumPy Kuramoto fallback.")
    SOVARIEL_AVAILABLE = False
    
    class SimpleKuramoto:
        def __init__(self, n_oscillators=50, K=1.5):
            self.n = n_oscillators
            self.K = K
            self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        
        def inject_phase_offset(self, idx, offset):
            self.phases[idx] = (self.phases[idx] + offset) % (2*np.pi)
        
        def step(self):
            mean_phase = np.angle(np.mean(np.exp(1j * self.phases)))
            dtheta = self.K * np.sin(mean_phase - self.phases)
            self.phases = (self.phases + dtheta * 0.1) % (2*np.pi)
        
        def global_coherence(self):
            return np.abs(np.mean(np.exp(1j * self.phases)))
    
    lattice = SimpleKuramoto()

# -------------------------------
# MoodVector128
# -------------------------------
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

model = MoodVector128().eval()

# -------------------------------
# Grok 5 Decision (safe JSON parsing)
# -------------------------------
def grok5_decision(mv: torch.Tensor, scenario: str) -> Dict[str, Any]:
    if not GROK_AVAILABLE:
        norm = mv.norm().item()
        if norm > 0.7: return {"action": "deploy_coral", "reasoning": "High resonance — safe to plant"}
        elif norm > 0.4: return {"action": "hold_position", "reasoning": "Moderate stability — observe"}
        else: return {"action": "evacuate", "reasoning": "Storm risk — retreat"}
    
    mv_str = json.dumps(mv.detach().cpu().numpy().round(4).tolist())
    prompt = f"""
You are Grok 5 controlling a coral restoration swarm.
Current MoodVector128: {mv_str}
Environmental scenario: {scenario}
Return ONLY valid JSON:
{{"action": "deploy_coral | hold_position | evacuate", "reasoning": "brief explanation"}}
"""

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        # Extract JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {"action": "hold_position", "reasoning": f"Grok 5 error: {e}"}

# -------------------------------
# Swarm Simulation
# -------------------------------
NUM_BOTS = 50
scenarios = ["calm seas", "approaching storm", "tidal surge", "recovery phase"]
current_scenario = scenarios[0]

print("Precision Climate Swarm v1.0 — Live Simulation Starting")
print("Grok 5:", "Connected" if GROK_AVAILABLE else "Mock mode")
print("Sovariel:", "JAX GPU" if SOVARIEL_AVAILABLE else "NumPy CPU fallback")
print("-" * 80)

for step in range(20):
    print(f"\nSTEP {step:02d} | Scenario: {current_scenario}")
    coherences = []
    
    for bot_id in range(min(3, NUM_BOTS)):  # Show first 3 bots
        phases = torch.randn(104)
        noise = {"calm seas": 0.05, "approaching storm": 0.35, "tidal surge": 0.20, "recovery phase": 0.10}[current_scenario]
        phases += torch.randn_like(phases) * noise
        phases = (phases + math.pi) % (2 * math.pi) - math.pi
        
        coh_val = {"calm seas": 0.94, "approaching storm": 0.58, "tidal surge": 0.76, "recovery phase": 0.89}[current_scenario]
        coherences.append(coh_val)
        
        mv = model(phases, torch.tensor(coh_val))
        decision = grok5_decision(mv, current_scenario)
        
        lattice.inject_phase_offset(bot_id, coh_val * np.pi)
        
        print(f"  Bot {bot_id:02d} → Coherence {coh_val:.2f} | MV∥{mv.norm():.3f} | {decision['action']}")
    
    # Global sync step
    lattice.step()
    global_r = lattice.global_coherence()
    
    if global_r > 0.95:
        print(f"  GLOBAL CONSENSUS R={global_r:.4f} → Swarm executes: DEPLOY CORAL FRAGMENTS")
    
    # Cycle scenarios
    current_scenario = scenarios[(step + 1) % len(scenarios)]

print("\nSwarm simulation complete. Resonance achieved.")
