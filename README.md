# ModularNN_AdaptiveHoneypot

Modular neural-network-driven honeypot framework built on Cowrie.

## Overview

This project combines a Cowrie SSH honeypot with reinforcement-learning components to support adaptive response policies. The codebase is organized to keep training logic, policy logging, and interaction tooling modular so behavior can be extended across different honeypot profiles.

## Repository Layout

- `cowrie/` - Cowrie source and runtime assets.
- `interactor/` - session-driving tools and command datasets.
- `run_interactor.py` - entrypoint for automated interaction runs.
- `start_cowrie.sh` - helper script to start or restart the local Cowrie instance.

## Quick Start

1. Set up Python virtual environments required by Cowrie and project tooling.
2. Start Cowrie:
   - `./start_cowrie.sh`
3. Run interactor sessions:
   - `python run_interactor.py`

## Notes

- Policy and session logs are written under Cowrie's `var/log/cowrie/`.
- Training and policy behavior are implemented under `cowrie/src/cowrie/rl/`.
