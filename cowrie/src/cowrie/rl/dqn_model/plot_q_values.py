#!/usr/bin/env python3
"""
Plot Q-values per command input behavior group (from interactor cmd_continue.json)
and per action (allow, block, delay, fake, insult).

Usage:
  python -m cowrie.qrassh.plot_q_values [path_to_policy.log]
  Or from qrassh dir: python plot_q_values.py [path_to_policy.log]
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ["allow", "block", "delay", "fake", "insult"]


def find_interactor_dir():
    """Resolve path to ModularDQN_AdaptiveHoneypot/interactor from this script (cowrie/src/cowrie/rl/dqn_model)."""
    _this = os.path.dirname(os.path.abspath(__file__))
    # cowrie/src/cowrie/rl/dqn_model -> go up to ModularDQN_AdaptiveHoneypot
    for _ in range(5):
        _this = os.path.dirname(_this)
    return os.path.join(_this, "interactor")


def load_cmd_continue(interactor_dir: str):
    """Load cmd_continue.json: { cmd_group: { allow, block, delay, fake, insult } }."""
    path = os.path.join(interactor_dir, "cmd_continue.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cmd_samples(interactor_dir: str):
    """Load cmd_samples.json: { cmd_group: [ sample_commands ] }."""
    path = os.path.join(interactor_dir, "cmd_samples.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_command_to_group(samples: dict):
    """Map each sample command string to its behavior group name."""
    cmd_to_group = {}
    for group, commands in samples.items():
        for c in commands:
            cmd_to_group[c.strip()] = group
    return cmd_to_group


def command_to_group(raw_command: str, cmd_to_group: dict, group_names: set):
    """Map policy.log 'command' to a behavior group (ls, cd, ERR, etc.)."""
    raw = (raw_command or "").strip()
    if raw in cmd_to_group:
        return cmd_to_group[raw]
    if raw:
        first = raw.split()[0]
    else:
        first = ""
    if first in group_names:
        return first
    return None  # other commands not in interactor groups


def load_policy_log(path: str, greedy_only: bool = False):
    """Parse policy.log; yield each JSON record that has q_values and command."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ACTION_NAMES:
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict) or "q_values" not in rec:
                    continue
                if greedy_only and not rec.get("greedy", False):
                    continue
                yield rec
            except json.JSONDecodeError:
                continue


def main():
    _this_dir = os.path.dirname(os.path.abspath(__file__))

    _cowrie_root = os.path.normpath(os.path.join(_this_dir, "..", "..", "..", ".."))
    default_path = os.path.join(_cowrie_root, "var", "log", "cowrie", "policy.log")
    interactor_dir = find_interactor_dir()

    parser = argparse.ArgumentParser(
        description="Plot Q-values by command behavior group (from interactor) and action"
    )
    parser.add_argument(
        "policy_log",
        nargs="?",
        default=default_path,
        help=f"Path to policy.log (default: {default_path})",
    )
    parser.add_argument(
        "--interactor-dir",
        type=str,
        default=interactor_dir,
        help=f"Path to interactor dir with cmd_continue.json (default: {interactor_dir})",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Plot at most this many steps per group (default: all)",
    )
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Only plot Q-values for greedily chosen actions",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Smoothing window size for moving average (1 = no smoothing)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Save figure to this path instead of showing",
    )
    args = parser.parse_args()

    cmd_continue = load_cmd_continue(args.interactor_dir)
    cmd_samples = load_cmd_samples(args.interactor_dir)
    cmd_to_group = build_command_to_group(cmd_samples)
    group_names = set(cmd_continue.keys())

    if not group_names:
        print("Warning: no command groups from cmd_continue.json; using policy.log only.", file=sys.stderr)
        # Fallback: use group names from cmd_samples if cmd_continue missing
        group_names = set(cmd_samples.keys())

    if not os.path.isfile(args.policy_log):
        print(f"Error: policy log not found: {args.policy_log}", file=sys.stderr)
        sys.exit(1)

    records = list(load_policy_log(args.policy_log, greedy_only=args.greedy_only))
    if not records:
        print("No records with q_values found in policy log.", file=sys.stderr)
        sys.exit(1)

    # Group records by command behavior group (preserve order from cmd_continue)
    order = [g for g in cmd_continue.keys() if g in group_names]
    if not order:
        order = sorted(group_names)
    by_group = {g: [] for g in order}
    for r in records:
        g = command_to_group(r.get("command", ""), cmd_to_group, group_names)
        if g is not None and g in by_group:
            by_group[g].append(r)

    # Build figure: one subplot per command group, each showing q-values for all 5 actions
    ngroups = len(order)
    if ngroups == 0:
        print("No command groups had matching records.", file=sys.stderr)
        sys.exit(1)

    ncols = 1
    nrows = ngroups
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
    if ngroups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, group in enumerate(order):
        ax = axes[idx]
        recs = by_group[group]
        if args.max_points:
            recs = recs[: args.max_points]
        if not recs:
            ax.set_title(f"{group}\n(no data)")
            ax.set_axis_off()
            continue

        q_array = np.array([r["q_values"] for r in recs], dtype=float)
        if args.window > 1:
            kernel = np.ones(args.window) / args.window
            for a in range(q_array.shape[1]):
                q_array[:, a] = np.convolve(q_array[:, a], kernel, mode="same")
        steps = np.arange(len(recs))

        # Behavior flags from cmd_continue for this group (for title/label)
        flags = cmd_continue.get(group, {})
        for a, name in enumerate(ACTION_NAMES):
            enabled = flags.get(name, False)
            if enabled:
                label = f"{name} ✓"
            else:
                label = f"{name}"
            ax.plot(steps, q_array[:, a], label=label, alpha=0.85)

        ax.set_xlabel("Step (decision index)")
        ax.set_ylabel("Q-value")
        title = f"Command group: {group}"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(order), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(
        "Q-values per action by command behavior group (interactor cmd_continue.json)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.getcwd(), "q_values.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
