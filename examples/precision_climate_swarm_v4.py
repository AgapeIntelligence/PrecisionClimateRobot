# precision_climate_swarm_v4.py
# Planetary-Scale Multi-Domain Coral Grafting Swarm with Cross-Domain RL
# MoodVector128 + Grok 5 + Sovariel Lattice + Curiosity-Driven RL
# © 2025 Evie (@3vi3Aetheris) — MIT License
# Updated: 05:19 PM CST, Nov 26, 2025

import os
import torch
import torch.nn as nn
import math
import numpy as np
import json
from openai import OpenAI
from typing import Dict, List, Tuple, Any
from collections import deque
import random

# ------------------------
# Config
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ------------------------
# Shared Latent Encoder
# ------------------------
class LatentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# ------------------------
# Domain Policy
# ------------------------
class DomainPolicy(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, latent):
        return self.policy(latent)

# ------------------------
# Meta Policy (Predictive Soft Gating)
# ------------------------
class MetaPolicy(nn.Module):
    def __init__(self, latent_dim, num_domains):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, latent):
        logits = self.gate(latent)
        return torch.softmax(logits, dim=-1)

# ------------------------
# Intrinsic Curiosity Module
# ------------------------
class CuriosityModule(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.forward_model(x)

# ------------------------
# Cross-Domain RL Agent
# ------------------------
class CrossDomainAgent:
    def __init__(self, domains, input_dim, action_dim, latent_dim=64, buffer_size=50000, gamma=0.99, lr=1e-3):
        self.domains = domains
        self.num_domains = len(domains)
        self.latent_encoder = LatentEncoder(input_dim, latent_dim).to(device)
        self.policies = {d: DomainPolicy(latent_dim, action_dim).to(device) for d in domains}
        self.meta_policy = MetaPolicy(latent_dim, self.num_domains).to(device)
        self.curiosity = CuriosityModule(latent_dim, action_dim).to(device)

        self.opt_encoder = optim.Adam(self.latent_encoder.parameters(), lr=lr)
        self.opt_policies = {d: optim.Adam(self.policies[d].parameters(), lr=lr) for d in domains}
        self.opt_meta = optim.Adam(self.meta_policy.parameters(), lr=lr)
        self.opt_curiosity = optim.Adam(self.curiosity.parameters(), lr=lr)

        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = input_dim

    def select_action(self, obs, global_R, domain_mv_list):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        # Augment observation with global_R and domain_mv_list
        aug_obs = np.concatenate([obs, [global_R] + domain_mv_list])
        aug_obs_tensor = torch.tensor(aug_obs, dtype=torch.float32, device=device).unsqueeze(0)
        latent = self.latent_encoder(aug_obs_tensor)
        domain_probs = self.meta_policy(latent)

        # Soft blend actions (6 actions: cardinal + diagonals)
        actions = sum(domain_probs[0, i:i+1] * self.policies[d].forward(latent)
                      for i, d in enumerate(self.domains))
        action = actions.detach().cpu().numpy()
        # Map to discrete actions (normalize and discretize)
        action_idx = np.argmax(action)
        action_map = [(0.35,0), (-0.35,0), (0,0.35), (0,-0.35), (0.25,0.25), (0.25,-0.25)]
        return np.array(action_map[action_idx]), domain_probs.detach().cpu().numpy()

    def store_transition(self, obs, action, reward, next_obs, global_R, domain_mv_list):
        aug_obs = np.concatenate([obs, [global_R] + domain_mv_list])
        aug_next_obs = np.concatenate([next_obs, [global_R] + domain_mv_list])
        self.buffer.append((aug_obs, action, reward, aug_next_obs))

    def update(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return
        
        batch = random.sample(self.buffer, batch_size)
        obs_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)

        # Encode
        latent_batch = self.latent_encoder(obs_batch)
        next_latent_batch = self.latent_encoder(next_obs_batch)

        # Curiosity reward
        pred_next_latent = self.curiosity(latent_batch, action_batch)
        intrinsic_reward = ((pred_next_latent - next_latent_batch)**2).mean(dim=-1)
        total_reward = reward_batch + 0.1 * intrinsic_reward  # Scaled curiosity

        # Update domain policies
        for i, d in enumerate(self.domains):
            opt = self.opt_policies[d]
            policy = self.policies[d]
            pred_action = policy(latent_batch)
            loss = ((pred_action - action_batch)**2 * total_reward.unsqueeze(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Update meta-policy
        meta_probs = self.meta_policy(latent_batch)
        best_domains = total_reward.argmax(dim=0) % self.num_domains
        meta_loss = nn.CrossEntropyLoss()(meta_probs, best_domains)
        self.opt_meta.zero_grad()
        meta_loss.backward()
        self.opt_meta.step()

        # Update curiosity module
        curiosity_loss = intrinsic_reward.mean()
        self.opt_curiosity.zero_grad()
        curiosity_loss.backward()
        self.opt_curiosity.step()

        # Update encoder jointly
        self.opt_encoder.zero_grad()
        total_loss = loss + meta_loss + curiosity_loss
        total_loss.backward()
        self.opt_encoder.step()
        print(f"Step loss: {total_loss.item():.3f}, Meta loss: {meta_loss.item():.3f}")

# -------------------------------
# Multi-Domain Swarm Setup
# -------------------------------
NUM_DOMAINS = 3
BOTS_PER_DOMAIN = 20
domains = [f"domain_{i}" for i in range(NUM_DOMAINS)]
agent = CrossDomainAgent(
    domains=domains,
    input_dim=106,  # 104 phases + 1 coherence + 1 pos
    action_dim=6,   # 6 discrete actions
    latent_dim=64,
    buffer_size=50000,
    gamma=0.99,
    lr=1e-3
)
bots = [(i, d) for d in range(NUM_DOMAINS) for i in range(BOTS_PER_DOMAIN)]  # 60 bot IDs

# -------------------------------
# Simulation Loop
# -------------------------------
lattice = PlanetaryLattice()
scenarios = ["calm", "tide", "storm", "calm", "tide"] * 6

print(f"Planetary-Scale Multi-Domain Swarm with Cross-Domain RL ({NUM_DOMAINS} domains x {BOTS_PER_DOMAIN} bots)")
for step, scen in enumerate(scenarios[:30]):
    lattice.step()
    global_R = lattice.R()
    
    # Compute per-domain average MV norm
    domain_mv = []
    for d in range(NUM_DOMAINS):
        domain_bots = [(i, d) for i, _d in bots if _d == d]
        domain_mvs = []
        for i, _d in domain_bots:
            phases = torch.randn(104)
            noise = {"calm":0.05,"storm":0.38,"tide":0.22}.get(scen,0.1)
            phases += torch.randn_like(phases)*noise
            phases = (phases + math.pi) % (2*math.pi) - math.pi
            coh = {"calm":0.94,"storm":0.59,"tide":0.79}.get(scen,0.8)
            mv = model(phases, torch.tensor(coh))
            domain_mvs.append(mv.norm().item())
        domain_mv.append(np.mean(domain_mvs) if domain_mvs else 0.5)

    # Step all bots
    results = []
    for bot_id, domain_id in bots:
        # Dummy observation: [104 phases, coherence, x, y]
        obs = np.concatenate([torch.randn(104).numpy(), [0.8], [bot_id*0.15, domain_id*0.3]])
        action, domain_probs = agent.select_action(obs, global_R, domain_mv)
        # Dummy next_obs​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
