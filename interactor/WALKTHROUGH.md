# Interactor walkthrough: how to start and use it

This guide explains how to run the **interactor**: the SSH client that drives the Cowrie honeypot so QRaSSH can learn from live sessions.

---

## 1. Prerequisites

- **Python 3** (same as Cowrie).
- **sshpass** (so the interactor can SSH with a password without a TTY):
  - macOS: `brew install sshpass` or install via [MacPorts](https://www.macports.org/).
  - Linux: `sudo apt install sshpass` (or your distro’s package).
- **Cowrie** set up under `ModularDQN_AdaptiveHoneypot/cowrie/` with the **QRaSSH** policy enabled (so each command is logged to `policy.log`).

---

## 2. Where to run from

All commands below assume you are in the **ModularDQN_AdaptiveHoneypot** project root:

```bash
cd /path/to/ModularDQN_AdaptiveHoneypot
```

The interactor expects:

- `cowrie/` (Cowrie install)
- `interactor/` (this package)
- `run_interactor.py` (entry script)
- `start_cowrie.sh` (used by the interactor to restart Cowrie)

---

## 3. Start Cowrie (optional)

You can start Cowrie yourself so it’s already running, or let the interactor do it.

**Option A – start Cowrie yourself:**

```bash
./start_cowrie.sh
```

Wait until Cowrie is listening on port **2222** (check logs in `cowrie/var/log/cowrie/`).

**Option B – let the interactor start it:**

When you run the interactor, it will **restart Cowrie** at startup (and create log/dirs if needed). You don’t need to run `start_cowrie.sh` first.

---

## 4. Run the interactor

From the **ModularDQN_AdaptiveHoneypot** root:

```bash
python run_interactor.py
```

This will:

1. Restart Cowrie (kill existing process and run `start_cowrie.sh`).
2. Load command samples from `interactor/cmd_samples.json` and preferences from `interactor/cmd_continue.json`.
3. Run **10 SSH sessions** by default (each session: login, send commands, then exit).
4. For each session, send a **random number of commands** (1 up to **10** per session by default).
5. After each command, read the last action from `cowrie/var/log/cowrie/policy.log`. If that action is not “continue” for the current state (see `cmd_continue.json`), the interactor sends `exit` and ends the session early.

Example output:

```
[*] Cowrie restarted
[*] Cowrie session id: b6414c30bc66
[*] [s1] Starting session (5 cmds), cowrie_session=b6414c30bc66
[*] [s1] [1/5] Sending: ls -a
[*] [s1] [1/5] Ack=True
[*] Last policy action: allow
[*] [s1] [2/5] Sending: wget http://example.com/file.txt
...
```

---

## 5. Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `-n`, `--num` | 10 | Number of SSH sessions to run. |
| `--max` | 10 | Maximum number of commands per session (actual per session is random 1..max). |
| `--sample_file` | `interactor/cmd_samples.json` | JSON: state → list of command strings to sample from. |
| `--continue_file` | `interactor/cmd_continue.json` | JSON: state → action → true/false (whether to continue the session after that action). |

**Examples:**

```bash
# Run 5 sessions, at most 3 commands per session
python run_interactor.py -n 5 --max 3

# Use custom sample/continue files
python run_interactor.py --sample_file my_cmds.json --continue_file my_continue.json
```

---

## 6. How the interactor uses the policy

- For each command it sends, **Cowrie + QRaSSH** decide an action (allow, block, delay, fake, insult) and append it to **`cowrie/var/log/cowrie/policy.log`**.
- The interactor reads the **last line** of `policy.log` (the plain action name, e.g. `allow`).
- It looks up in **`cmd_continue.json`** whether to continue the session for the current **state** (e.g. `ls`, `wget`) and that **action**.
- If the value is `false`, it sends `exit` and closes the session; otherwise it continues to the next command.

So `cmd_continue.json` controls “after this action on this state, do we keep sending commands or end the session?”

---

## 7. Config files (quick reference)

- **`interactor/cmd_samples.json`**  
  Maps **state** (e.g. `"ls"`, `"wget"`) to a list of **command strings**. The interactor picks a random state, then a random command from that list.

- **`interactor/cmd_continue.json`**  
  Maps **state** → **action** → `true`/`false`.  
  Example: `"ls"` → `"allow"` → `true` means “if the policy chose allow for a command in state ls, continue the session.”

---

## 8. Logs and outputs

| Path | Description |
|------|-------------|
| `cowrie/var/log/cowrie/policy.log` | QRaSSH decisions: JSON lines (command, action, q_values, greedy, …) plus a short action line. The interactor reads the last line for the current action. |
| `cowrie/var/log/cowrie/cowrie.json` | Cowrie events (logins, commands, etc.). The interactor uses this to know when a command was received (ack). |
| `cowrie/var/log/cowrie/cowrie.log` | Cowrie text log. |

---

## 9. Plot Q-values from the policy log

After running the interactor (and/or real traffic), you can plot Q-values from `policy.log`:

```bash
# From repo root; uses default policy log path
python -m honeypot_rl.dqn_model.plot_q_values

# Custom log path or save figure
python -m honeypot_rl.dqn_model.plot_q_values cowrie/var/log/cowrie/policy.log -o q_values.png
```

Requires **matplotlib** (`pip install matplotlib` or use Cowrie’s venv).

---

## 10. Minimal “run once” checklist

1. `cd` to **ModularDQN_AdaptiveHoneypot**.
2. (Optional) Install **sshpass** if needed.
3. Run:  
   `python run_interactor.py`
4. Optionally run more sessions:  
   `python run_interactor.py -n 20 --max 5`
5. Inspect `cowrie/var/log/cowrie/policy.log` or run `plot_q_values` as above.

That’s the full walkthrough for starting and using the interactor.
