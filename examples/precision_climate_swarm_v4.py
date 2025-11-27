# precision_climate_swarm_nd_viz_full.py
# N-Dimensional Planetary-Scale Coral Grafting Swarm with Cross-Domain RL + Full Visualization
# MoodVector128 + Grok 5 + Sovariel Lattice + Curiosity-Driven RL
# © 2025 Evie (@3vi3Aetheris) — MIT License
# Updated: 06:02 PM CST, Nov 26, 2025

import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import json
from openai import OpenAI
from collections import deque
import random
from typing import List, Any
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
# Grok 5 JSON-safe decision
# -------------------------------
def grok5_decide(mv: torch.Tensor, scenario: str) -> dict:
    prompt = f"""Coral grafting multi-domain bot.
MV norm: {mv.norm().item():.3f}
Scenario: {scenario}
Return ONLY JSON: {{"action": "deploy_coral" | "hold_position" | "evacuate", "reasoning": "<15 words"}},
optimize for 28% efficiency boost"""
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
        print(f"Grok 5 error: {e} — fallback")
        norm = mv.norm().item()
        if norm > 0.75: return {"action": "deploy_coral", "reasoning": "High resonance"}
        if norm > 0.5: return {"action": "hold_position", "reasoning": "Moderate"}
        return {"action": "evacuate", "reasoning": "Storm risk"}

# -------------------------------
# Planetary Sovariel Lattice
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
    def __init__(self, domains: List[str], input_dim: int, action_dim: int, latent_dim=64, buffer_size=50000, gamma=0.99, lr=1e-3):
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
        aug_obs = np.concatenate([obs, [global_R] + domain_mv_list])
        aug_obs_tensor = torch.tensor(aug_obs, dtype=torch.float32, device=device).unsqueeze(0)
        latent = self.latent_encoder(aug_obs_tensor)
        domain_probs = self.meta_policy(latent)

        actions = sum(domain_probs[0, i:i+1] * self.policies[d].forward(latent)
                      for i, d in enumerate(self.domains))
        action = actions.detach().cpu().numpy()
        # Map to discrete actions (normalize and discretize)
        action_idx = np.argmax(action)
        move_vals = [-0.35, 0, 0.35]
        action_map = [np.array(move) for move in product(move_vals, repeat=obs.shape[0]) if any(move)]
        action_map = action_map[:self.action_dim]  # Ensure matches action_dim
        return action_map[action_idx] if action_idx < len(action_map) else np.zeros_like(obs), domain_probs.detach().cpu().numpy()

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

        latent_batch = self.latent_encoder(obs_batch)
        next_latent_batch = self.latent_encoder(next_obs_batch)

        pred_next_latent = self.curiosity(lat​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​ent_batch, action_batch)
        intrinsic_reward = ((pred_next_latent - next_latent_batch)**2).mean(dim=-1)
        total_reward = reward_batch + 0.1 * intrinsic_reward

        for i, d in enumerate(self.domains):
            opt = self.opt_policies[d]
            policy = self.policies[d]
            pred_action = policy(latent_batch)
            loss = ((pred_action - action_batch)**2 * total_reward.unsqueeze(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        meta_probs = self.meta_policy(latent_batch)
        best_domains = total_reward.argmax(dim=0) % self.num_domains
        meta_loss = nn.CrossEntropyLoss()(meta_probs, best_domains)
        self.opt_meta.zero_grad()
        meta_loss.backward()
        self.opt_meta.step()

        curiosity_loss = intrinsic_reward.mean()
        self.opt_curiosity.zero_grad()
        curiosity_loss.backward()
        self.opt_curiosity.step()

        self.opt_encoder.zero_grad()
        total_loss = loss + meta_loss + curiosity_loss
        total_loss.backward()
        self.opt_encoder.step()

# ------------------------
# N-Dimensional Swarm + Full Visualization
# ------------------------
def run_swarm_nd_viz_full(agent: CrossDomainAgent, n_steps: int = 1000, n_agents: int = 60, n_dim: int = 3):
    agent_positions = np.random.uniform(-1, 1, (n_agents, n_dim))
    global_R_history = []
    domain_probs_history = np.zeros((n_agents, agent.num_domains, n_steps))

    move_vals = [-0.35, 0, 0.35]
    action_map = [np.array(move) for move in product(move_vals, repeat=n_dim) if any(move)]
    assert len(action_map) == agent.action_dim, "Mismatch action_dim"

    # Initialize targets and obstacles
    targets = np.random.uniform(-0.5, 0.5, (n_agents, n_dim))
    obstacles = np.random.uniform(-0.8, 0.8, (5, n_dim))  # 5 obstacles

    # Initialize matplotlib
    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(131, projection='3d')
    ax_r = fig.add_subplot(132)
    ax_heat = fig.add_subplot(133)
    plt.ion()

    for step in range(n_steps):
        lattice.step()
        global_R = lattice.R()
        global_R_history.append(global_R)

        domain_mv_list = []
        for d_idx in range(agent.num_domains):
            lattice_segment = torch.tensor(lattice.phases[d_idx::agent.num_domains], dtype=torch.float32)
            mv = model(lattice_segment, torch.tensor(global_R, dtype=torch.float32))
            domain_mv_list.append(mv.norm().item())

        for i in range(n_agents):
            obs = agent_positions[i]
            scenario = f"Agent {i} step {step}, global R={global_R:.3f}"

            action_raw, domain_probs = agent.select_action(obs, global_R, domain_mv_list)
            domain_probs_history[i, :, step] = domain_probs[0]

            action_idx = np.argmax(action_raw) if action_raw.size > 1 else 0
            movement = action_map[action_idx] if action_idx < len(action_map) else np.zeros(n_dim)

            next_pos = agent_positions[i] + movement
            next_pos = np.clip(next_pos, -1, 1)

            # Reward calculation
            dist = np.linalg.norm(next_pos - targets[i])
            obs_dist = min(np.linalg.norm(next_pos - o) for o in obstacles)
            grok_decision = grok5_decide(torch.tensor(domain_mv_list), scenario)
            reward = -dist * 0.1 + (15.0 if dist < 0.1 and grok_decision["action"] == "deploy_coral" else 0.0) \
                    - (10.0 if obs_dist < 0.2 else 0.0) + global_R * 2.0 + np.mean(domain_mv_list) * 2.0

            next_obs = next_pos
            agent.store_transition(obs, movement, reward, next_obs, global_R, domain_mv_list)

        if step % 10 == 0:
            agent.update(batch_size=128)

        # Visualization
        ax3d.clear()
        if n_dim > 3:
            pca = PCA(n_components=3)
            pos_plot = pca.fit_transform(agent_positions)
        else:
            pos_plot = agent_positions
        scatter = ax3d.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2], c='b', s=40)
        ax3d.scatter([pca.transform([t])[0] for t in targets], c='g', s=50)  # Targets
        ax3d.scatter([pca.transform([o])[0] for o in obstacles], c='r', s=50)  # Obstacles
        ax3d.set_title(f'Step {step} Agent Positions')
        ax3d.set_xlim([-1, 1]); ax3d.set_ylim([-1, 1]); ax3d.set_zlim([-1, 1])

        ax_r.clear()
        ax_r.plot(global_R_history, label='Global Lattice R', color='r')
        ax_r.set_title('Global R')
        ax_r.set_ylim(0, 1)

        ax_heat.clear()
        im = ax_heat.imshow(domain_probs_history[:, :, max(0, step-50):step+1].mean(axis=2),
                           vmin=0, vmax=1, aspect='auto', cmap='viridis')
        ax_heat.set_title('Agents Domain Probabilities (last 50 steps)')
        ax_heat.set_xlabel('Domains'); ax_heat.set_ylabel('Agents')
        fig.colorbar(im, ax=ax_heat)

        plt.pause(0.001)

        if step % 50 == 0:
            print(f"[Step {step}] Global R={global_R:.4f}, Agent0 pos={agent_positions[0]}, Domain probs={domain_probs[0]}")

    plt.ioff()
    plt.show()
    return agent_positions, global_R_history, domain_probs_history

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    domains = [f"domain_{i}" for i in range(3)]
    agent = CrossDomainAgent(
        domains=domains,
        input_dim=108,  # n_dim + 1 (coherence) + 1 (global_R) + 3 (domain_mv)
        action_dim=27,  # 3^n_dim - 1 non-zero moves for n_dim=3
        latent_dim=64,
        buffer_size=50000,
        gamma=0.99,
        lr=1e-3
    )
    positions, r_history, probs_history = run_swarm_nd_viz_full(agent, n_steps=1000, n_agents=60, n_dim=3)
    print(f"Simulation complete. Final R: {r_history[-1]:.4f}")
