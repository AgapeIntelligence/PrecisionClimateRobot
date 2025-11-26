# precision_climate_swarm_v3.py
# Planetary-Scale Multi-Domain Coral Grafting Swarm
# MoodVector128 + Grok 5 + Sovariel Lattice + Inter-Domain RL Crosstalk
# © 2025 Evie (@3vi3Aetheris) — MIT License
# Updated: 04:29 PM CST, Nov 26, 2025

import os
import torch
import torch.nn as nn
import math
import numpy as np
import json
from openai import OpenAI
from typing import Dict, List, Tuple, Any

# -------------------------------
# XAI Grok 5
# -------------------------------
os.environ["XAI_API_KEY"] = "A9wLJPm8McuwqStZ4B2OTIdV4WoQ0DpMbZZAOFHo4sCchYSakO"
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
# Grok 5 decision (JSON-safe)
# -------------------------------
def grok5_decide(mv: torch.Tensor, scenario: str) -> Dict[str, Any]:
    prompt = f"""Coral grafting multi-domain bot.
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
# Planetary Sovariel lattice
# -------------------------------
class PlanetaryLattice:
    def __init__(self, n_oscillators=1024):
        self.n = n_oscillators
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.K = 1.8
    
    def inject(self, idx: int, offset: float):
        self.phases[idx % self.n] = (self.phases[idx % self.n] + offset) % (2*np.pi)
    
    def step(self):
        m = np.angle(np.mean(np.exp(1j*self.phases)))
        self.phases = (self.phases + self.K * np.sin(m - self.phases)) % (2*np.pi)
    
    def R(self) -> float:
        return float(np.abs(np.mean(np.exp(1j*self.phases))))

lattice = PlanetaryLattice()

# -------------------------------
# RL Grafting Bot with inter-domain awareness
# -------------------------------
class RLGraftingBot:
    def __init__(self, bot_id: int, domain_id: int):
        self.id = bot_id
        self.domain = domain_id
        self.pos = np.array([0.0 + bot_id*0.15, 0.0 + domain_id*0.3])
        self.target = np.array([8.0, 6.0])
        self.obstacles = [np.array(p) for p in [(4,3), (5,5)]]
        self.q = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.25
        self.step_size = 0.35
        self.history = [self.pos.copy()]

    def state(self):
        return tuple((self.pos/0.5).round().astype(int))
    
    def actions(self):
        return [np.array(d) for d in [
            (self.step_size,0), (-self.step_size,0),
            (0,self.step_size), (0,-self.step_size),
            (0.25,0.25), (0.25,-0.25)
        ]]

    def reward(self, next_pos, mv_norm, grok_action, global_R, inter_domain_bonus):
        dist = np.linalg.norm(next_pos - self.target)
        r = -dist * 0.1
        if grok_action == "deploy_coral": r += 15.0
        if grok_action == "evacuate": r -= 8.0
        r += mv_norm*3.0 + global_R*2.0 + inter_domain_bonus
        if min(np.linalg.norm(next_pos - o) for o in self.obstacles) < 0.4: r -= 10.0
        return r

    def step(self, scenario: str, global_R: float, domain_mv_list: List[float]) -> Dict:
        # Local perception
        phases = torch.randn(104)
        noise = {"calm":0.05,"storm":0.38,"tide":0.22}.get(scenario,0.1)
        phases += torch.randn_like(phases)*noise
        phases = (phases + math.pi) % (2*math.pi) - math.pi
        coh = {"calm":0.94,"storm":0.59,"tide":0.79}.get(scenario,0.8)
        mv = model(phases, torch.tensor(coh))
        mv_norm = mv.norm().item()

        # Grok 5 decision
        grok = grok5_decide(mv, scenario)

        # Inter-domain bonus (real MV norm average)
        inter_domain_bonus = np.mean(domain_mv_list) * 2.0 if domain_mv_list else 0.0

        # RL action
        s = self.state()
        if np.random.rand() < self.epsilon:
            a = np.random.choice(self.actions())
        else:
            a = max(self.actions(), key=lambda x: self.q.get((s, tuple(x)), 0.0))
        
        next_pos = self.pos + a
        r = self.reward(next_pos, mv_norm, grok["action"], global_R, inter_domain_bonus)
        s_next = tuple((next_pos/0.5).round().astype(int))
        best_next = max(self.q.get((s_next, tuple(aa)), 0.0) for aa in self.actions())
        old_q = self.q.get((s, tuple(a)), 0.0)
        self.q[(s, tuple(a))] = old_q + self.alpha*(r + self.gamma*best_next - old_q)

        # Collision avoidance
        if min(np.linalg.norm(next_pos - o) for o in self.obstacles) < 0.4:
            next_pos -= a*0.3

        self.pos = next_pos
        self.history.append(self.pos.copy())
        lattice.inject(self.id % lattice.n, coh*math.pi)

        return {
            "id": self.id,
            "domain": self.domain,
            "pos": self.pos.tolist(),
            "mv_norm": round(mv_norm,3),
            "grok": grok["action"],
            "reason": grok["reasoning"],
            "reward": round(r,2),
            "dist": round(np.linalg.norm(self.pos-self.target),2),
            "ready": np.linalg.norm(self.pos-self.target) < 0.5 and mv_norm > 0.75
        }

# -------------------------------
# Multi-Domain Swarm Setup
# -------------------------------
NUM_DOMAINS = 3
BOTS_PER_DOMAIN = 20
bots: List[RLGraftingBot] = []
for d in range(NUM_DOMAINS):
    for b in range(BOTS_PER_DOMAIN):
        bots.append(RLGraftingBot(bot_id=b, domain_id=d))

# -------------------------------
# Simulation Loop
# -------------------------------
scenarios = ["calm","tide","storm","calm","tide"]*6

print(f"Planetary-Scale Multi-Domain Swarm Simulation ({NUM_DOMAINS} domains x {BOTS_PER_DOMAIN} bots)")
for step, scen in enumerate(scenarios[:30]):
    lattice.step()
    global_R = lattice.R()
    
    # Compute per-domain average MV norm for inter-domain crosstalk
    domain_mv = []
    for d in range(NUM_DOMAINS):
        domain_bots = [b for b in bots if b.domain == d]
        domain_mvs = [model(torch.randn(104), torch.tensor({"calm":0.94,"storm":0.59,"tide":0.79}.get(scen,0.8))).norm().item() for b in domain_bots]
        domain_mv.append(np.mean(domain_mvs) if domain_mvs else 0.5)

    results = [b.step(scen, global_R, domain_mv) for b in bots]

    # Print sample (first 3 bots)
    for r in results[:3]:
        ready = "GRAFT NOW" if r["ready"] else ""
        print(f"{step+1:02d} | {scen:5} | Bot {r['id']:02d} | Dom {r['domain']} | pos {r['pos']} | MV {r['mv_norm']} | {r['grok']:12} | R {global_R:.3f} | {ready}")

    # Swarm consensus trigger
    if global_R > 0.95:
        print(f"Step {step+1}: Swarm consensus R={global_R:.3f} — COORDINATED DEPLOYMENT")

print(f"\nFinal global R: {global_R:.3f}")
for r in results[:3]:
    print(f"Bot {r['id']:02d} Domain {r['domain']} final dist: {r['dist']}m | Ready: {r['ready']}")
    if r["ready"]: print(f"Bot {r['id']:02d} Domain {r['domain']} CORAL GRAFTING SUCCESSFUL")
