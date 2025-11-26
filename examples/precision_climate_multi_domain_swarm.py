# precision_climate_multi_domain_swarm.py
# Multi-Domain Precision Climate Swarm — MoodVector128 + Grok 5 + Sovariel + RL
# Supports reef, tree, glacier, and environmental sensing
# © 2025 Evie (@3vi3Aetheris) — MIT License

import os
import torch
import torch.nn as nn
import math
import numpy as np
import json
from openai import OpenAI
from typing import Dict, List, Tuple, Any

# -------------------------------
# XAI Grok 5 (your API key included)
# -------------------------------
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
# Grok 5 decision (JSON safe)
# -------------------------------
def grok5_decide(mv: torch.Tensor, scenario: str, domain: str) -> Dict[str, Any]:
    prompt = f"""
Precision climate swarm bot — {domain}
MV norm: {mv.norm().item():.3f}
Scenario: {scenario}
Return ONLY JSON: {{"action": "deploy | hold_position | evacuate", "reasoning": "<15 words"}}"""
    try:
        resp = client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        j = json.loads(resp.choices[0].message.content.strip().split("```")[0])
        return j
    except Exception as e:
        # fallback
        norm = mv.norm().item()
        if norm > 0.75: return {"action": "deploy", "reasoning": "High resonance"}
        if norm > 0.5: return {"action": "hold_position", "reasoning": "Moderate"}
        return {"action": "evacuate", "reasoning": "Storm risk"}

# -------------------------------
# Sovariel-style lattice
# -------------------------------
class Lattice:
    def __init__(self, n:int):
        self.phases = np.random.uniform(0, 2*np.pi, n)
    def inject(self, idx:int, offset:float): 
        self.phases[idx] = (self.phases[idx] + offset) % (2*np.pi)
    def step(self):
        m = np.angle(np.mean(np.exp(1j*self.phases)))
        self.phases = (self.phases + 1.8*np.sin(m - self.phases)) % (2*np.pi)
    def R(self): return np.abs(np.mean(np.exp(1j*self.phases)))

# -------------------------------
# RL Agent
# -------------------------------
class RLGraftingBot:
    def __init__(self, bot_id:int, domain:str, start_pos=(0,0), target_pos=(8,6), obstacles=[(4,3),(5,5)]):
        self.id = bot_id
        self.domain = domain
        self.pos = np.array(start_pos) + np.array([bot_id*0.1, 0.0])
        self.target = np.array(target_pos)
        self.obstacles = [np.array(o) for o in obstacles]
        self.q = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.25
        self.step_size = 0.35
        self.history = [self.pos.copy()]

    def state(self): return tuple((self.pos/0.5).round().astype(int))
    def actions(self): 
        # 6 directions
        return [np.array(d) for d in [(0.35,0), (-0.35,0), (0,0.35), (0,-0.35), (0.25,0.25), (0.25,-0.25)]]

    def reward(self, next_pos, mv_norm, grok_action, global_R):
        dist = np.linalg.norm(next_pos - self.target)
        r = -dist * 0.1
        if grok_action == "deploy": r += 15.0
        if grok_action == "evacuate": r -= 8.0
        r += mv_norm*3.0 + global_R*2.0
        if min(np.linalg.norm(next_pos-o) for o in self.obstacles) < 0.4: r -= 10.0
        return r

    def step(self, scenario:str, global_R:float):
        # Local phases
        phases = torch.randn(104)
        noise = {"reef":0.22, "tree":0.18, "glacier":0.25, "env":0.15}.get(self.domain,0.2)
        phases += torch.randn_like(phases)*noise
        phases = (phases + math.pi) % (2*math.pi) - math.pi
        coh = {"reef":0.79, "tree":0.85, "glacier":0.81, "env":0.88}.get(self.domain,0.8)
        mv = model(phases, torch.tensor(coh))
        mv_norm = mv.norm().item()

        # Grok decision
        grok = grok5_decide(mv, scenario, self.domain)

        # RL action
        s = self.state()
        if np.random.rand()<self.epsilon: a=np.random.choice(self.actions())
        else: a=max(self.actions(), key=lambda x:self.q.get((s, tuple(x)),0.0))

        next_pos = self.pos + a
        r = self.reward(next_pos, mv_norm, grok["action"], global_R)
        s_next = self.state()

        best_next = max(self.q.get((s_next, tuple(aa)),0.0) for aa in self.actions())
        old_q = self.q.get((s, tuple(a)),0.0)
        self.q[(s, tuple(a))] = old_q + self.alpha*(r+self.gamma*best_next-old_q)

        # Collision back-off
        if min(np.linalg.norm(next_pos-o) for o in self.obstacles)<0.4:
            next_pos -= a*0.3

        self.pos = next_pos
        self.history.append(self.pos.copy())
        return {
            "id":self.id,
            "domain":self.domain,
            "pos":self.pos.tolist(),
            "mv_norm":round(mv_norm,3),
            "grok":grok["action"],
            "reason":grok["reasoning"],
            "reward":round(r,2),
            "dist":round(np.linalg.norm(self.pos-self.target),2),
            "ready":np.linalg.norm(self.pos-self.target)<0.5 and mv_norm>0.75
        }

# -------------------------------
# Swarm Simulation
# -------------------------------
DOMAINS = ["reef","tree","glacier","env"]
NUM_PER_DOMAIN = 25  # adjustable
swarm = [RLGraftingBot(i + d*NUM_PER_DOMAIN, d) for d in DOMAINS for i in range(NUM_PER_DOMAIN)]
lattice = Lattice(n=len(swarm))

scenarios = ["calm","tide","storm","calm","env_shift"]*5

print(f"Multi-domain precision swarm — {len(swarm)} bots active")
for step, scen in enumerate(scenarios[:30]):
    lattice.step()
    global_R = lattice.R()
    results = [bot.step(scen, global_R) for bot in swarm]

    # Print first 4 bots for readability
    for res in results[:4]:
        ready = "READY" if res["ready"] else ""
        print(f"{step+1:02d} | {scen:8} | Bot {res['id']:02d} ({res['domain']}) | Pos {res['pos']} | MV {res['mv_norm']} | {res['grok']} | R {global_R:.3f} | {ready}")

    if global_R>0.95:
        print(f"Step {step+1}: Swarm consensus R={global_R:.3f} — COORDINATED DEPLOYMENT")

print(f"\nFinal global R: {global_R:.3f}")
for res in results[:4]:
    print(f"Bot {res['id']:02d} ({res['domain']}) final dist: {res['dist']} | Ready: {res['ready']}")
