# Dual-Arm Panda Handover with RL using SAC — Extension to panda-gym

A single-agent reinforcement learning environment for a **two-arm cube handover task**
built on top of [panda-gym](https://github.com/qgallouedec/panda-gym) and trained with
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## Project overview

Two Franka Panda robots are mounted side-by-side on a wide table.
The **left arm** must pick up a cube from the left side of the table and
carry it to a central **handover zone**.
The **right arm** must then pick it up and place it on a randomised **target region**
on the right side.

A single policy controls both arms simultaneously (8-dimensional continuous action space).
The environment follows the standard GoalEnv / Gymnasium API and is compatible with
Stable-Baselines3 out of the box.

```
Top-down view
─────────────
         y
    +0.3 │  ● right arm base    [target region]
         │
    0.0  │              [handover zone]
         │
   -0.3  │  ● left arm base     [cube start]
         └──────────────────────────────> x
           -0.6        -0.1  0.0
```

---

## Files added / modified

| File | Role |
|------|------|
| `panda_gym/envs/robots/dual_panda_robot.py` | `DualPanda` wrapper — combines two `Panda` instances into one action/obs interface |
| `panda_gym/envs/tasks/dual_panda_task.py` | `HandoverTask` — scene creation, reward, success detection |
| `panda_gym/envs/dual_panda_env.py` | `DualPandaHandoverEnv` — glues robot + task into a Gymnasium env |
| `panda_gym/__init__.py` | Registers `PandaDualHandover-v0` with gymnasium |
| `examples/train_dual_panda.py` | SAC training script with checkpointing and eval callback |
| `examples/eval_dual_panda.py` | Evaluation script with per-episode and aggregate stats |

---

## Spaces

| | Shape | Range | Description |
|---|---|---|---|
| `action_space` | `(8,)` | `[-1, 1]` | `[dx_L, dy_L, dz_L, grip_L, dx_R, dy_R, dz_R, grip_R]` |
| `obs["observation"]` | `(26,)` | unbounded | Both EE positions, velocities, gripper widths + cube pose/vel |
| `obs["achieved_goal"]` | `(3,)` | — | Current cube position |
| `obs["desired_goal"]` | `(3,)` | — | Target position (randomised each episode) |

---

## Installation

**Requirements:** Python 3.8, a working panda-gym repo clone, and pip.

```bash
# 1. Clone and enter the repo
git clone https://github.com/qgallouedec/panda-gym
cd panda-gym

# 2. Create and activate a virtual environment
python3 -m venv .panda_venv
source .panda_venv/bin/activate

# 3. Install panda-gym in editable mode
pip install -e .

# 4. Install CPU-only PyTorch (avoids the 800 MB CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install Stable-Baselines3 and TensorBoard
pip install stable-baselines3 tensorboard
```

Verified working with:
- Python 3.8.10
- panda-gym 3.0.8
- stable-baselines3 2.4.1
- torch 2.4.1+cpu
- gymnasium 1.0.0

---

## Running

### Sanity check (no training, random policy)

```python
import gymnasium as gym
import panda_gym

env = gym.make("PandaDualHandover-v0")
obs, info = env.reset()
for _ in range(50):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
env.close()
```

### Train

```bash
# Default: 1 000 000 steps, logs to logs/, models to models/
python examples/train_dual_panda.py

# Custom
python examples/train_dual_panda.py --timesteps 2000000 --run-name exp1

# Monitor training
tensorboard --logdir logs/
```

### Evaluate a saved model

```bash
python examples/eval_dual_panda.py \
    --model models/<run_name>/best_model/best_model \
    --episodes 50

# With rendering
python examples/eval_dual_panda.py \
    --model models/<run_name>/best_model/best_model \
    --episodes 10 --render
```

Reported metrics:
- **Success rate** — cube reached target within 5 cm
- **Handover rate** — cube crossed the central handover zone (stage 1 proxy)
- **Mean ± std reward**
- **Mean episode length**

---

## Known limitations

### Environment / physics

1. **No real handover mechanics.** The environment does not enforce that the right arm
   physically grasps the cube from the left arm mid-air. In practice, the left arm drops
   the cube near the handover zone and the right arm re-grasps from the table. This is a
   realistic simplification but not a true in-air transfer.

2. **Gravity and IK may cause arm collisions.** Both arms share a table but there is no
   self-collision avoidance between them. At random initialisations they can clip through
   each other. Mitigate with conservative base offsets or an explicit collision penalty.

3. **Fixed handover zone.** The handover zone position is constant across episodes. A
   harder generalisation task would randomise it.

4. **PyBullet IK drift.** The Panda EE controller uses analytical IK that can drift from
   desired Cartesian positions over long episodes, especially near joint limits. Increase
   `sim_freq` or add joint-limit penalties if this is a problem.

### Reward / training

5. **HER incompatible.** `HandoverTask.compute_reward()` queries live EE positions from
   the simulation, making it incompatible with vectorised Hindsight Experience Replay goal
   relabelling. If you want HER, remove the shaping term and rely on `–d(cube, target)` only.

6. **Sparse credit assignment.** The right arm only gets reward signal once the cube has
   been moved by the left arm. Early in training, both arms may learn to do nothing.
   Increasing `shaping_weight` (default 0.3) can help if training stalls.

7. **Single-environment training.** The training script uses one env instance. For faster
   wall-clock training, wrap with `make_vec_env` (but note SAC is not designed for
   vectorised envs — use PPO + `SubprocVecEnv` if you want parallelism).

8. **CPU-only torch.** The install above uses the CPU wheel. Training will be slow (~3–5 h
   for 1 M steps on a laptop). Use a GPU build of torch if a CUDA device is available.

### Scope

9. **Single-agent formulation.** Both arms are controlled by one policy. A multi-agent
   formulation (e.g. MADDPG) would be more principled for true cooperative tasks but is
   significantly more complex.

10. **No domain randomisation.** Friction, mass, and arm kinematics are fixed. A sim-to-real
    deployment would need randomisation and a calibrated URDF.

---

## Debugging checklist

Use this list when the environment misbehaves or training does not converge.

### Environment / import

- [ ] `import panda_gym` completes without error — confirms registration is loaded
- [ ] `gym.make("PandaDualHandover-v0")` succeeds — confirms entry_point resolves
- [ ] `env.reset()` returns obs with keys `observation`, `achieved_goal`, `desired_goal`
- [ ] `obs["observation"].shape == (26,)` and `action_space.shape == (8,)`
- [ ] Random-action loop runs 200 steps without crashing

### Physics / scene

- [ ] Run with `render_mode="human"` and confirm both arms appear on the table
- [ ] Confirm the cube spawns on the **left** side (y ≈ −0.15)
- [ ] Confirm the orange target marker appears on the **right** side (y ≈ +0.20)
- [ ] Arms are not visually intersecting at reset
- [ ] Cube does not fall through the table (check `TABLE_HEIGHT` matches URDF)

### Reward signal

- [ ] `reward` is negative at every step for a random policy (expected: −0.2 to −0.5)
- [ ] `reward` increases when cube is manually moved toward target (print reward each step)
- [ ] `info["is_success"]` becomes `True` when cube is placed within 5 cm of target
- [ ] Disable shaping (`shaping_weight=0.0`) and confirm reward = `−d(cube, target)`

### Training

- [ ] TensorBoard shows `rollout/ep_rew_mean` increasing after ~50 k steps
- [ ] `rollout/ep_len_mean` is not stuck at `max_episode_steps` (200) from step 1
      — if it is, the policy is learning to time-out, not to succeed
- [ ] Eval callback reports non-zero success within ~300 k steps; if not, try:
      - Increase `shaping_weight` to 0.5
      - Decrease `learning_starts` to 1 000
      - Increase `batch_size` to 1024
- [ ] Check for NaN in rewards: `assert not np.isnan(reward)` inside the step loop
- [ ] Model checkpoint files exist in `models/<run_name>/` after 50 k steps

### Evaluation

- [ ] `eval_dual_panda.py --model <path>` loads without error
- [ ] `handover_rate` > `success_rate` (left arm's job is easier than the full task)
- [ ] With `--render`, arms are visibly attempting to move toward the cube
