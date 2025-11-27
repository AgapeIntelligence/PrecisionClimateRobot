# precision_climate_swarm_final.py
# Ultimate N-D Planetary Coral Swarm
# Multi-Agent Audio + LSTM Threat Prediction + Attention + Cross-Domain RL + Swarm Communication
# Hierarchical Curiosity + Energy-Aware Rewards + Sovariel Lattice + MoodVector128
# © 2025 Evie (@3vi3Aetheris)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import json
from openai import OpenAI
from collections import deque
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import torchaudio
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Grok 5 API
os.environ["XAI_API_KEY"] = "A9wLJPm8McuwqStZ4B2OTIdV4WoQ0DpMbZZAOFHo4sCchYSakO"
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

# -------------------------------
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

# -------------------------------
# Planetary Sovariel Lattice
class PlanetaryLattice:
    def __init__(self, n_oscillators=1024):
        self.n = n_oscillators
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.K = 1.8
    def step(self):
        m = np.angle(np.mean(np.exp(1j*self.phases)))
        self.phases = (self.phases + self.K*np.sin(m-self.phases)) % (2*np.pi)
    def R(self) -> float:
        return float(np.abs(np.mean(np.exp(1j*self.phases))))

lattice = PlanetaryLattice()
mood_model = MoodVector128().to(device)

# -------------------------------
# Audio Features + Grok Alerts
def extract_audio_features(waveform, sample_rate, n_mels=64):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    log_mel = torch.log1p(mel_spec)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.mean(dim=-1)

def grok5_audio_alert(features: torch.Tensor, context: str) -> dict:
    prompt = f"""Live wildlife audio monitoring.
Features norm: {features.norm().item():.3f}
Context: {context}
Return ONLY JSON: {{"alert": "none" | "potential_threat" | "species_detected", "reasoning": "<20 words"}}"""
    try:
        resp = client.chat.completions.create(model="grok-beta",
                                             messages=[{"role": "user", "content": prompt}],
                                             temperature=0.0, max_tokens=80)
        j = json.loads(resp.choices[0].message.content.strip().split("```")[0])
        return j
    except:
        norm = features.norm().item()
        if norm > 1.2: return {"alert": "potential_threat", "reasoning": "High energy signature"}
        if norm > 0.7: return {"alert": "species_detected", "reasoning": "Moderate signature"}
        return {"alert": "none", "reasoning": "Quiet environment"}

# -------------------------------
# Threat Memory & LSTM Predictor
class ThreatMemory:
    def __init__(self, decay=0.9): self.decay = decay; self.memory = 0.0
    def update(self, current_alert_strength):
        self.memory = self.memory * self.decay + current_alert_strength * (1 - self.decay)
        return self.memory

class ThreatPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):  # x: batch, seq_len, features
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return torch.sigmoid(pred)

# -------------------------------
# Latent Encoder, Domain + Meta Policy + Curiosity + Attention
class LatentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self, x): return self.encoder(x)

class DomainPolicy(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, latent): return self.policy(latent)

class AttentionMetaPolicy(nn.Module):
    def __init__(self, latent_dim, num_domains):
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, num_domains)
    def forward(self, latent, comm_latents=None):
        q = self.query(latent)
        k = self.key(latent)
        v = self.value(latent)
        if comm_latents is not None:
            k = torch.cat([k, comm_latents], dim=0)
            v = torch.cat([v, comm_latents[:, :self.value.out_features]], dim=0)
        attn = torch.softmax(q @ k.T / math.sqrt(latent.shape[-1]), dim=-1)
        return torch.softmax(attn @ v, dim=-1)

class CuriosityModule(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.forward_model = nn.Sequential(nn.Linear(latent_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self, latent, action): return self.forward_model(torch.cat([latent, action], dim=-1))

# -------------------------------
# Cross-Domain Agent
class CrossDomainAgent:
    def __init__(self, domains, input_dim, action_dim, latent_dim=64, buffer_size=50000, lr=1e-3):
        self.domains = domains; self.num_domains = len(domains)
        self.latent_encoder = LatentEncoder(input_dim, latent_dim).to(device)
        self.policies = {d: DomainPolicy(latent_dim, action_dim).to(device) for d in domains}
        self.meta_policy = AttentionMetaPolicy(latent_dim, self.num_domains).to(device)
        self.curiosity = CuriosityModule(latent_dim, action_dim).to(device)
        self.opt_encoder = optim.Adam(self.latent_encoder.parameters(), lr=lr)
        self.opt_policies = {d: optim.Adam(self.policies[d].parameters(), lr=lr) for d in domains}
        self.opt_meta = optim.Adam(self.meta_policy.parameters(), lr=lr)
        self.opt_curiosity = optim.Adam(self.curiosity.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.latent_dim = latent_dim; self.action_dim = action_dim; self.input_dim = input_dim
        self.model = MoodVector128().to(device)

    def select_action(self, obs, global_R, domain_mv_list, audio_strength=0.0, predicted_threat=0.0, comm_latents=None):
        aug_obs = np.concatenate([obs, [global_R] + domain_mv_list + [audio_strength, predicted_threat]])
        aug_obs_tensor = torch.tensor(aug_obs, dtype=torch.float32, device=device).unsqueeze(0)
        latent = self.latent_encoder(aug_obs_tensor)
        domain_probs = self.meta_policy(latent, comm_latents)
        if audio_strength > 0.5 or predicted_threat > 0.5:
            domain_probs[0, -1] += max(audio_strength, predicted_threat)
            domain_probs = domain_probs / domain_probs.sum()
        actions = sum(domain_probs[0, i:i+1] * self.policies[d].forward(latent) for i, d in enumerate(self.domains))
        return actions.detach().cpu().numpy(), domain_probs.detach().cpu().numpy(), latent

    def store_transition(self, obs, action, reward, next_obs, global_R, domain_mv_list):
        aug_obs = np.concatenate([obs, [global_R] + domain_mv_list])
        aug_next_obs = np.concatenate([next_obs, [global_R] + domain_mv_list])
        self.buffer.append((aug_obs, action, reward, aug_next_obs))

    def update(self, batch_size=128):
        if len(self.buffer) < batch_size: return
        batch = random.sample(self.buffer, batch_size)
        obs_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        latent_batch = self.latent_encoder(obs_batch)
        next_latent_batch = self.latent_encoder(next_obs_batch)
        pred_next_latent = self.curiosity(latent_batch, action_batch)
        intrinsic_reward = ((pred_next_latent - next_latent_batch)**2).mean(dim=-1)
        total_reward = reward_batch + 0.1 * intrinsic_reward
        for i, d in enumerate(self.domains):
            opt = self.opt_policies[d]; policy = self.policies[d]
            pred_action = policy(latent_batch)
            loss = ((pred_action - action_batch)**2 * total_reward.unsqueeze(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        meta_probs = self.meta_policy(latent_batch)
        best_domains = total_reward.argmax(dim=0) % self.num_domains
        meta_loss = nn.CrossEntropyLoss()(meta_probs, best_domains)
        self.opt_meta.zero_grad(); meta_loss.backward(); self.opt_meta.step()
        curiosity_loss = intrinsic_reward.mean()
        self.opt_curiosity.zero_grad(); curiosity_loss.backward(); self.opt_curiosity.step()
        self.opt_encoder.zero_grad()
        total_loss = loss + meta_loss + curiosity_loss
        total_loss.backward()
        self.opt_encoder.step()

# -------------------------------
# Run Swarm with Full Inter-Agent Communication
def run_swarm_final(agent, lattice, n_steps=500, n_agents=60, n_dim=3):
    agent_positions = np.random.uniform(-1, 1, (n_agents, n_dim))
    global_R_history = []
    move_vals = [-0.35, 0, 0.35]
    action_map = [np.array(m) for m in product(move_vals, repeat=n_dim) if any(m)]
    assert len(action_map) == agent.action_dim, f"Action dim mismatch: expected {len(action_map)}, got {agent.action_dim}"
    threat_memory = ThreatMemory(decay=0.9)
    predictor = ThreatPredictor(input_dim=2).to(device)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
    predictor_buffer = deque(maxlen=20)
    # Simulate audio (replace with real data)
    audio_files = ["sample1.wav", "sample2.wav"]  # Placeholder
    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(131, projection='3d')
    ax_r = fig.add_subplot(132)
    ax_threat = fig.add_subplot(133)
    plt.ion()

    for step in range(n_steps):
        lattice.step()
        global_R = lattice.R()
        global_R_history.append(global_R)
        domain_mv_list = [agent.model(torch.tensor(lattice.phases[d::agent.num_domains], dtype=torch.float32),
                                     torch.tensor(global_R, dtype=torch.float32)).norm().item()
                          for d in range(agent.num_domains)]
        # Audio processing
        audio_strengths = []
        for i in range(n_agents):
            waveform, sample_rate = torchaudio.load(audio_files[i % len(audio_files)])  # Cyclic placeholder
            features = extract_audio_features(waveform, sample_rate)
            alert = grok5_audio_alert(features, f"Agent {i} at step {step}")
            strength = 0.0 if alert["alert"] == "none" else (0.5 if alert["alert"] == "species_detected" else 1.0)
            audio_strengths.append(threat_memory.update(strength))
        predictor_buffer.append([global_R, np.mean(audio_strengths)])
        if len(predictor_buffer) >= 5:
            lstm_input = torch.tensor(np.array(predictor_buffer)[None, :, :], dtype=torch.float32, device=device)
            predicted_threat = predictor(lstm_input).item()
            # Update predictor
            target = torch.tensor([max(audio_strengths)], dtype=torch.float32, device=device)
            loss = nn.MSELoss()(predictor(lstm_input), target)
            predictor_optimizer.zero_grad()
            loss.backward()
            predictor_optimizer.step()
        else:
            predicted_threat = 0.0
        comm_latents = torch.cat([agent.latent_encoder(torch.tensor(np.concatenate([agent_positions[i],
                                                                                  [global_R] + domain_mv_list +
                                                                                  [audio_strengths[i], predicted_threat]]),
                                                              dtype=torch.float32, device=device).unsqueeze(0))
                                  for i in range(n_agents)], dim=0)
        for i in range(n_agents):
            obs = agent_positions[i]
            action_raw, domain_probs, latent = agent.select_action(obs, global_R, domain_mv_list,
                                                                  audio_strengths[i], predicted_threat, comm_latents)
            action_idx = np.argmax(action_raw)
            movement = action_map[action_idx] if action_idx < len(action_map) else np.zeros(n_dim)
            next_pos = agent_positions[i] + movement
            next_pos = np.clip(next_pos, -1, 1)
            # Energy-aware reward
            energy_cost = np.linalg.norm(movement) * 0.1
            reward = global_R - (global_R_history[-2] if step > 0 else 0.0) - energy_cost + \
                     (0.5 if audio_strengths[i] > 0.5 else 0.0) * (1 - predicted_threat)
            agent.store_transition(obs, movement, reward, next_pos, global_R, domain_mv_list)
        if step % 10 == 0:
            agent.update(batch_size=128)
        # Visualization
        ax3d.clear()
        pos_plot = PCA(n_components=3).fit_transform(agent_positions) if n_dim > 3 else agent_positions
        ax3d.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2], c='b', s=40)
        ax3d.set_title(f'Step {step} Agent Positions')
        ax3d.set_xlim([-1, 1]); ax3d.set_ylim([-1, 1]); ax3d.set_zlim([-1, 1])
        ax_r.clear()
        ax_r.plot(global_R_history, color='r')
        ax_r.set_title('Global Lattice Coherence R')
        ax_r.set_ylim(0, 1)
        ax_threat.clear()
        ax_threat.plot([b[1] for b in predictor_buffer], label='Threat Memory')
        ax_threat.plot([predicted_threat] * len(predictor_buffer), label='Predicted Threat')
        ax_threat.set_title('Threat Levels')
        ax_threat.legend()
        plt.pause(0.001)
        if step % 50 == 0:
            print(f"[Step {step}] Global R={global_R:.4f}, Agent0 pos={agent_positions[0]}, Threat={predicted_threat:.3f}")
    plt.ioff()
    plt.show()
    return agent_positions, global_R_history

# -------------------------------
# Example Usage
if __name__ == "__main__":
    domains = ["ocean", "reef", "air"]
    agent = CrossDomainAgent(domains, input_dim=3 + 1 + len(domains) + 2, action_dim=26)  # 3^3-1 for 3D
    positions, R_hist = run_swarm_final(agent, lattice, n_steps=1000, n_agents=60, n_dim=3)
    print(f"Simulation complete. Final R: {R_hist[-1]:.4f}")
