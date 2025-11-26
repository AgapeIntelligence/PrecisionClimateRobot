# precision_climate_swarm_rl_final.py
# Full RL coral grafting swarm — MoodVector128 + Grok 5 + Sovariel + Q-learning
# © 2025 Evie (@3vi3Aetheris) — MIT License
# Run with: export XAI_API_KEY=your_key or use env var below

import os
import torch
import torch.nn as nn
import math
import numpy as np
import json
from openai import OpenAI
from typing import Dict, List, Tuple, Any

# -------------------------------
# XAI Grok 5 (real, with your key)
# -------------------------------
# Set your API key via environment variable (recommended)
# os.environ["XAI_API_KEY"] = "vxGmXellwp581pE4Z6RNDdYB1Afnykl9asTzn9J6s6X09QXt1b"
# For this run, using env var directly (securely handle in production)
os.environ["XAI_API_KEY"] = "vxGmXellwp581pE4Z6RNDdYB1Afnykl9asTzn9J6s6X09QXt1b"
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

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
# Grok 5 decision (real + safe JSON)
# -------------------------------
def grok5_decide(mv: torch.Tensor, scenario: str) -> Dict[str, Any]:
    prompt = f"""Coral grafting swarm bot.
MV norm: {mv.norm().item():.3f}
Scenario: {scenario}
Return ONLY JSON: {{"action": "deploy_coral" | "hold_position" | "evacuate", "reasoning": "<15 words"}}"""
    try:
        resp = client.chat.completions.create(
            model="grok-beta", messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=80
        )
        j = json.loads(resp.choices[0].message.content.strip().split("```")[0])
        return j
    except Exception as e:
        print(f"Grok 5 error: {e} — fallback")
        norm = mv.norm().item()
        if norm > 0.75: return {"action": "deploy_coral", "reasoning": "High resonance"}
        if norm > 0.5: return {"action": "hold_position", "reasoning": "Moderate"}
        return {"action": "evacuate", "reasoning": "Storm risk"}

# -------------------------------
# Simple Sovariel-style lattice (NumPy)
# -------------------------------
class Lattice:
    def __init__(self, n=50): self.phases = np.random.uniform(0, 2*np.pi, n)
    def inject(self, idx, offset): self.phases[idx] = (self.phases[idx] + offset) % (2*np.pi)
    def step(self): 
        m = np.angle(np.mean(np.exp(1j*self.phases)))
        self.phases = (self.phases + 1.8*np.sin(m - self.phases)) % (2*np.pi)
    def R(self): return np.abs(np.mean(np.exp(1j*self.phases)))

lattice = Lattice()

# -------------------------------
# RL Pathing Agent (Q-learning + Grok 5)
# -------------------------------
class RLGraftingBot:
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.target = np.array([8.0, 6.0])
        self.obstacles = [np.array(p) for p in [(4,3), (5,5)]]
        self.q = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.25
        self.step_size = 0.35
        self.history = [self.pos.copy()]

    def state(self): return tuple(((self.pos)/0.5).round().astype(int))
    def actions(self): return [np.array(d) for d in [(0.35,0), (-0.35,0), (0,0.35), (0,-0.35), (0.25,0.25), (0.25,-0.25), (-0.25,0.25), (-0.25,-0.25)]]

    def reward(self, next_pos, mv_norm, grok_action):
        dist = np.linalg.norm(next_pos - self.target)
        r = -dist * 0.1
        if grok_action == "deploy_coral": r += 15.0
        if grok_action == "evacuate": r -= 8.0
        r += mv_norm * 3.0
        if min(np.linalg.norm(next_pos - o) for o in self.obstacles) < 0.4: r -= 10.0
        return r

    def step(self, scenario: str) -> Dict:
        # Local perception
        phases = torch.randn(104)
        phases += torch.randn_like(phases) * {"calm":0.05, "storm":0.38, "tide":0.22}.get(scenario, 0.1)
        phases = (phases + math.pi) % (2*math.pi) - math.pi
        coh = {"calm":0.94, "storm":0.59, "tide":0.79}.get(scenario, 0.8)
        mv = model(phases, torch.tensor(coh))
        mv_norm = mv.norm().item()

        # Grok 5 decision
        grok = grok5_decide(mv, scenario)

        # RL action
        s = self.state()
        if np.random.rand() < self.epsilon:
            a = np.random.choice(self.actions())
        else:
            a = max(self.actions(), key=lambda x: self.q.get((s, tuple(x)), 0.0))

        next_pos = self.pos + a
        r = self.reward(next_pos, mv_norm, grok["action"])
        s_next = self.state()

        # Q-update
        best_next = max(self.q.get((s_next, tuple(aa)), 0.0) for aa in self.actions())
        old_q = self.q.get((s, tuple(a)), 0.0)
        self.q[(s, tuple(a))] = old_q + self.alpha * (r + self.gamma * best_next - old_q)

        # Collision back-off
        if min(np.linalg.norm(next_pos - o) for o in self.obstacles) < 0.4:
            next_pos -= a * 0.3

        self.pos = next_pos
        self.history.append(self.pos.copy())
        lattice.inject(0, coh * np.pi)
        lattice.step()

        return {
            "pos": self.pos.tolist(),
            "mv_norm": round(mv_norm, 3),
            "grok": grok["action"],
            "reason": grok["reasoning"],
            "reward": round(r, 2),
            "dist": round(np.linalg.norm(self.pos - self.target), 2),
            "global_R": round(lattice.R(), 3),
            "ready": np.linalg.norm(self.pos - self.target) < 0.5 and mv_norm > 0.75
        }

# -------------------------------
# Run the swarm bot
# -------------------------------
bot = RLGraftingBot()
scenarios = ["calm","calm","tide","storm","calm","tide","calm","calm","storm","calm"] * 3

print("RL + Grok 5 Coral Grafting Swarm — LIVE (03:59 PM CST, Nov 26, 2025)")
for i, scen in enumerate(scenarios[:30]):
    res = bot.step(scen)
    ready = "GRAFT NOW" if res["ready"] else ""
    print(f"{i+1:02d} | {scen:5} | pos {res['pos']} | MV {res['mv_norm']} | {res['grok']:12} | R {res['global_R']:.3f} | {ready}")

print(f"\nFinal distance: {res['dist']}m | Steps taken: {len(bot.history)} | Global R: {res['global_R']:.3f}")
if res["ready"]: print("CORAL GRAFTING SUCCESSFUL")
