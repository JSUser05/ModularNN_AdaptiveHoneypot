# ModularNN_AdaptiveHoneypot

Modular neural-network-driven honeypot framework built on Cowrie.

## Overview

This project combines a Cowrie SSH honeypot with reinforcement-learning components to support adaptive response policies. The codebase is organized to keep training logic, policy logging, and interaction tooling modular so behavior can be extended across different honeypot profiles.

## Repository Layout

- `honeypot_rl/` - **Reinforcement learning code** (DQN agent, replay buffer, rewards, policy logging, plotting). Kept at the repo root so it is easy to review separately from the Cowrie tree.
- `cowrie/` - Cowrie source and runtime assets (shell/protocol hooks import `honeypot_rl`).
- `interactor/` - **not** in the remote repo (gitignored); keep your own copy locally for session-driving tools and command datasets.
- `run_interactor.py` - entrypoint for automated interaction runs (expects a local `interactor/` package).
- `start_cowrie.sh` - helper script to start or restart the local Cowrie instance.

## Quick Start

1. Set up Python virtual environments required by Cowrie and project tooling.
2. Start Cowrie:
   - `./start_cowrie.sh`
3. Run interactor sessions:
   - `python run_interactor.py`

## Notes

- Policy and session logs are written under `cowrie/var/log/cowrie/`.
- `start_cowrie.sh` and `cowrie/bin/cowrie` set `PYTHONPATH` to include this **repository root** so `import honeypot_rl` resolves at runtime.
- Plot Q-values from policy logs:  
  `python -m honeypot_rl.dqn_model.plot_q_values`
