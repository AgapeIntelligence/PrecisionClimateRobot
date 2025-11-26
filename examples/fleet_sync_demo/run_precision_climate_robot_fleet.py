# examples/fleet_sync_demo/run_precision_climate_robot_fleet.py

import numpy as np
import torch

# Local perception brain
from precision_climate_robot import PrecisionClimateBrain

# Sovariel global synchrony layer
from sovariel import JAXLiveAudioLattice, mars_fleet_lock


# --------------------------------------------------------
# 1. Initialize Precision Climate Robot brain (per-bot)
# --------------------------------------------------------
sensors_spec = {
    'mic': 200,
    'imu': 200,
    'depth': 10,
}

brain = PrecisionClimateBrain(sensors_spec)


# --------------------------------------------------------
# 2. Initialize Sovariel Kuramoto lattice (fleet-level sync)
# --------------------------------------------------------

# Example: 1000 climate robots operating in forest grid
n_bots = 1000
lattice = JAXLiveAudioLattice(n_oscillators=n_bots)   # GPU/TPU optimized Kuramoto system


# --------------------------------------------------------
# 3. One simulation step
# --------------------------------------------------------
def sim_step(bot_id: int, sensors_window: dict):
    """
    One step for bot_id:
    - Local perception  → COHERENCE + ACTION
    - Inject local coherence into the Kuramoto lattice
    - Lattice steps globally
    - Global synchrony R influences whether the bot executes action
    """
    # ----- LOCAL PERCEPTION -----
    local_result = brain.compute(sensors_window)
    actions = local_result["action"]

    vx, vy, dz = actions["vx"], actions["vy"], actions["dz"]
    local_c = local_result["coherence"]

    # ----- UPDATE GLOBAL LATTICE -----
    # Convert local coherence (0..1) → phase offset (0..π) 
    lattice.inject_phase_offset(bot_id, float(local_c) * np.pi)

    # Step the global synchrony
    lattice.step()
    global_r = lattice.global_coherence()  # 0..1

    # ----- FLEET CONSENSUS GATE -----
    if global_r > 0.95:
        print(
            f"[EXECUTE] Bot {bot_id} | R={global_r:.4f} | vx={vx:.2f} vy={vy:.2f} dz={dz:.2f}"
        )
    else:
        print(f"[HOLD] Bot {bot_id} | R={global_r:.4f} | coherence={local_c:.3f}")

    return {
        'local_action': actions,
        'local_coherence': local_c,
        'global_r': global_r,
    }


# --------------------------------------------------------
# 4. Demo Loop (Synthetic sensors)
# --------------------------------------------------------
if __name__ == "__main__":
    T = 200  # 1 sec @ 200 Hz
    t = np.linspace(0, 1.0, T, endpoint=False)

    for step in range(10):
        # Synthetic environmental signals
        sensors = {
            'mic': np.sin(2*np.pi*6*t) + 0.05*np.random.randn(T),
            'imu': 0.1*np.sin(2*np.pi*12*t) + 0.02*np.random.randn(T),
            'depth': np.convolve(np.random.randn(10), np.ones(5)/5, mode='same') * 0.05
        }

        result = sim_step(bot_id=0, sensors_window=sensors)

        print(
            f"Step {step:02d} | "
            f"Precision={result['local_action']['precision_score']:.3f} | "
            f"Global R={result['global_r']:.4f}"
        )

    # Example for planetary-scale fleets:
    # mars_lattice = mars_fleet_lock(n_ships=42)   # Delay-tolerant, relativistic-safe
