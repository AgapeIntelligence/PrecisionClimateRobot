# precision_climate_swarm_full.py
# Multi-Bot Precision Climate Robotics Swarm — RL + MoodVector128 + Grok5 + Sovariel Lattice
# © 2025 Evie (@3vi3Aetheris)

import os, math, json
import numpy as np
import torch
import torch.nn as nn
from openai import OpenAI

# -------------------------------
# XAI Client
# -------------------------------
os.environ["XAI_API_KEY"] = "vxGmXeIIwp581pE4Z6RNDdYB1Afnykl9asT6Xo9QXt1b"
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
# Grok5 Decision
# -------------------------------
def grok5_decision(mv: torch.Tensor, scenario: str) -> dict:
    mv_list = mv.detach().cpu().numpy().round(4).tolist()
    prompt = f"""
MoodVector128: {mv_list}
Scenario: {scenario}
Return JSON only: {{"action": "deploy_coral | hold_position | evacuate", "reasoning": "brief"}}
"""
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=80
    )
    content = response.choices[0].message.content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    return json.loads(content[start:end])

# -------------------------------
# Sovariel Lattice (Kuramoto-style)
# -------------------------------
class SovarielLattice:
    def __init__(self, n_oscillators):
        self.N = n_oscillators
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.K = 1.5
    def inject_phase_offset(self, idx, offset):
        self.phases[idx] = (self.phases[idx] + offset) % (2*np.pi)
    def step(self):
        mean_phase = np.angle(np.mean(np.exp(1j * self.phases)))
        self.phases += self.K * np.sin(mean_phase - self.phases) * 0.1
        self.phases %= 2*np.pi
    def global_coherence(self):
        return np.abs(np.mean(np.exp(1j * self.phases)))

# -------------------------------
# RL Pathing Agent
# -------------------------------
class RLClimateBot:
    def __init__(self, bot_id, start=(0,0), target=(8,6), obstacles=[(4,3),(5,5)]):
        self.id = bot_id
        self.pos = np.array(start, dtype=float)
        self.target = np.array(target, dtype=float)
        self.obstacles = [np.array(o) for o in obstacles]
        self.q_table = {}
        self.step_size = 0.3
        self.precision_threshold = 0.5
        self.path_history = [self.pos.copy()]
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.3
    def discretize(self):
        return tuple((self.pos / self.step_size).round().astype(int))
    def actions(self):
        return [np.array([dx,dy]) for dx,dy in [(0.3,0),(0.3,0),(0,-0.3),(0,0.3)]]
    def reward(self, next_pos, mv_norm, graft_ready):
        dist = np.linalg.norm(next_pos - self.target)
        reward = -dist*0.1
        if graft_ready: reward += 10.0
        reward += mv_norm*2.0
        min_obs = min(np.linalg.norm(next_pos - o) for o in self.obstacles)
        if min_obs < 0.3: reward -= 5.0
        return reward
    def step(self, scenario, lattice:SovarielLattice):
        # Generate synthetic phases
        phases = torch.randn(104)
        noise = {"calm":0.05,"storm":0.35,"tide":0.2}[scenario]
        phases += torch.randn_like(phases) * noise
        coh = {"calm":0.95,"storm":0.62,"tide":0.81}[scenario]
        mv = model(phases, torch.tensor(coh))
        mv_norm = mv.norm().item()
        # Grok5
        decision = grok5_decision(mv, scenario)
        # Inject local coherence into lattice
        lattice.inject_phase_offset(self.id, coh*np.pi)
        lattice.step()
        global_r = lattice.global_coherence()
        # RL action selection
        state = self.discretize()
        if np.random.rand()<self.epsilon:
            act = self.actions()[np.random.randint(4)]
        else:
            act = max(self.actions(), key=lambda a:self.q_table.get((state,tuple(a)),0.0))
        next_pos = self.pos + act
        min_dist = min(np.linalg.norm(next_pos - o) for o in self.obstacles)
        if min_dist<0.3: next_pos -= act*0.2
        graft_ready = np.linalg.norm(next_pos - self.target)<self.precision_threshold and mv_norm>0.7
        # Q-learning
        r = self.reward(next_pos, mv_norm, graft_ready)
        next_state = tuple((next_pos/self.step_size).round().astype(int))
        old_q = self.q_table.get((state,tuple(act)),0.0)
        max_next_q = max([self.q_table.get((next_state,tuple(a)),0.0) for a in self.actions()])
        new_q = old_q + self.lr*(r+self.gamma*max_next_q-old_q)
        self.q_table[(state,tuple(act))]=new_q
        self.pos = next_pos
        self.path_history.append(self.pos.copy())
        return {"pos":self.pos.tolist(),"mv_norm":mv_norm,"graft_ready":graft_ready,
                "action":act.tolist(),"reward":r,"reasoning":decision.get("reasoning",""),"global_r":global_r}

# -------------------------------
# SWARM SIMULATION
# -------------------------------
NUM_BOTS=6
bots = [RLClimateBot(i) for i in range(NUM_BOTS)]
lattice = SovarielLattice(NUM_BOTS)
scenarios = ["calm","storm","calm","tide"]*5

print("=== Multi-Bot Precision Climate Swarm Simulation ===")
for step, scenario in enumerate(scenarios):
    for bot in bots:
        result = bot.step(scenario,lattice)
        print(f"Step {step}, Bot {bot.id}, Pos {result['pos']}, GraftReady {result['graft_ready']}, MV {result['mv_norm']:.3f}, Global R {result['global_r']:.3f}, Reasoning: {result['reasoning']}")
print("\nSimulation complete.")
