# Microduck simulation training: session handoff

Read this first in a new **local** session on the user's Windows PC. This is a side project, separate from
VisionLabeler: do not change any VisionLabeler file. Full guide with the same commands:
https://claude.ai/artifact/Cjq9oBu6ZVwobyz1TXjv5A

## 1. Goal

Train Pollen Robotics' **Microduck** (a 25 cm walking duck robot, $399, preorder only; the user has no robot yet)
to walk **in simulation** on this PC, then drive the trained duck with the keyboard in the simulator. Training code:
official [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (mjlab + MuJoCo Warp + PPO).

## 2. How to work with the user

- **Ask permission before each phase** (their explicit request). Before each phase say in a few lines: what it does,
  what it changes on the PC, how to undo it. Wait for "yes".
- Short, simple, concrete answers; the user is not a native English speaker.
- Things only the user can do: click "Yes" on Windows admin (UAC) prompts, restart the PC, create the Ubuntu
  username/password.

## 3. The PC

Windows 11, NVIDIA RTX 4070 SUPER 12 GB (from docs/HANDOFF.md). VisionLabeler normally uses the GPU:
**run `stop.bat` in the VisionLabeler folder before training** (ask first). Need about 20 GB free disk.
Set Windows sleep to Never while training.

## 4. Phases (ask before each)

Nothing has been run on the PC yet. Phase 1 has not started.

**Phase 1: WSL2 + Ubuntu, check the GPU.** First check what exists: `wsl --status` and `wsl -l -v`.
Update the Windows NVIDIA driver (never install an NVIDIA driver inside Ubuntu). Then, in an admin PowerShell:

```powershell
wsl --install -d Ubuntu      # or `wsl --update` if WSL is already there
```

Restart if asked; the user creates the Ubuntu user. Then inside Ubuntu:

```bash
nvidia-smi                                   # must show the RTX 4070 SUPER
sudo apt update && sudo apt install -y git curl
```

Undo: `wsl --unregister Ubuntu`.

**Phase 2: download the training code** (several GB, 10-30 min). Keep it in the Ubuntu home, not `/mnt/c` (slow).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
mkdir -p ~/duck && cd ~/duck
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
export UV_HTTP_TIMEOUT=600
uv sync
uv run list-envs                             # must list Mjlab-Velocity-Flat-MicroDuck
```

**Phase 3: quick tests.** A) Pollen's trained duck in the simulator (proves the 3D window works).
B) a 5-iteration training run (proves GPU training works).

```bash
cd ~/duck/microduck_rl
mkdir -p ~/duck/policies
for f in alpha_walking.onnx alpha_stand.onnx; do curl -fsSL -o ~/duck/policies/$f https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/$f; done
export GALLIUM_DRIVER=d3d12
uv run scripts/infer_policy.py --walking ~/duck/policies/alpha_walking.onnx --standing ~/duck/policies/alpha_stand.onnx --new-cmd-obs

uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard --agent.run-name smoke
```

Simulator keys are typed in the **Ubuntu terminal**, not the 3D window: up/down speed, A/E turn, Space stop,
P push, Q quit. `infer_policy.py` reads keys with Linux `termios`, so it must run in WSL, not native Windows Python.

**Phase 4: full training** (estimate about 4 h on this GPU; not measured). Keep the Ubuntu window open.

```bash
cd ~/duck/microduck_rl
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard --agent.run-name walk1
```

Progress (second Ubuntu window): `cd ~/duck/microduck_rl && uvx tensorboard --logdir logs`, then
http://localhost:6006 in the Windows browser. Resume after a stop (replace 1500 with the last checkpoint):

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.logger tensorboard --agent.run-name walk1b --agent.resume True --agent.load-run '.*_walk1' --agent.load-checkpoint model_1500.pt
```

**Phase 5: export and drive the user's own duck** (works mid-training too).

```bash
cd ~/duck/microduck_rl
CK=$(ls -t logs/rsl_rl/velocity/*_walk1/model_*.pt | head -1); echo $CK
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint-file $CK --onnx-file ~/duck/policies/my_walk.onnx
export GALLIUM_DRIVER=d3d12
uv run scripts/infer_policy.py --walking ~/duck/policies/my_walk.onnx --new-cmd-obs
```

## 5. Facts checked against the code (microduck_rl `b02fdea`, 2026-10-12; mjlab 1.3.0)

- Python must be 3.12 (`requires-python >=3.12,<3.13`); uv installs it.
- The walking task's default `max_iterations` is **50,000** (days). Always pass `--agent.max-iterations`.
- mjlab's default logger is `wandb` (needs an account); `--agent.logger tensorboard` keeps it local.
- Checkpoints: `logs/rsl_rl/velocity/<datetime>_<run-name>/model_<iter>.pt`, every 250 iterations.
- `--agent.load-checkpoint` picks alphabetically when given a pattern, so give an exact file name.
- Always export with `scripts/export.py` (it bakes the observation normalizer into the ONNX).

## 6. Known problems and fixes

| Problem | Fix |
| --- | --- |
| `nvidia-smi` fails in Ubuntu | Update the Windows driver, `wsl --update`, `wsl --shutdown`, reopen |
| `uv sync` timeout | `export UV_HTTP_TIMEOUT=600`, run again |
| CUDA out of memory | `--env.scene.num-envs 2048` (or 1024); check VisionLabeler is stopped |
| Training died overnight | WSL shuts down when no window is attached, or the PC slept; resume (Phase 4) |
| 3D window choppy | `export GALLIUM_DRIVER=d3d12` |
| Pollen's default `velstand.onnx` will not walk | Use `alpha_walking.onnx` + `alpha_stand.onnx`, or add `--no-bam` |

Optional later: the unofficial [CdyDoodle/maxy_duck](https://github.com/CdyDoodle/maxy_duck) wraps these steps
with a native Windows viewer and a training monitor (messages in Chinese; read its scripts before running them).
