import json
import os
import sys
from typing import Optional, List

# honeypot_rl/dqn_model -> repo root -> cowrie/
_this_dir = os.path.dirname(os.path.abspath(__file__))
cowrie_root = os.path.normpath(os.path.join(_this_dir, "..", "..", "cowrie"))
POLICY_LOG_PATH = os.path.join(cowrie_root, "var", "log", "cowrie", "policy.log")


def set_policy_log_path(path: str) -> None:
    """Override for tests; normal runs use POLICY_LOG_PATH."""
    global POLICY_LOG_PATH
    POLICY_LOG_PATH = path


def get_policy_log_path() -> str:
    return POLICY_LOG_PATH


def write_policy_decision(
    session_id: str,
    command: str,
    action_id: int,
    action_name: str,
    q_values: Optional[List[float]] = None,
    greedy: bool = True,
) -> None:
    """Append one JSON line to policy.log at cowrie/var/log/cowrie/policy.log. Dir is created if missing."""
    path = POLICY_LOG_PATH
    obj = {
        "session_id": session_id,
        "command": command,
        "action": action_name,
        "action_id": action_id,
        "greedy": greedy,
    }
    if q_values is not None:
        obj["q_values"] = [float(x) for x in q_values]
    line = json.dumps(obj) + "\n"
    try:
        dirpath = os.path.dirname(path)
        os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except OSError as e:
        print(f"[QRaSSH] policy.log write failed: path={path!r} err={e}", file=sys.stderr)
