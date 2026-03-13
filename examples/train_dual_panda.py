"""train_dual_panda.py

Train a SAC policy on the dual-arm PandaDualHandover-v0 environment.

Algorithm choice — SAC over PPO
────────────────────────────────
* SAC is off-policy (replay buffer), so every transition is re-used many
  times.  Robotic manipulation tasks are sample-hungry; SAC typically
  reaches the same performance as PPO in 3–5× fewer environment steps.
* The dense reward and continuous 8-DoF action space both suit SAC's
  maximum-entropy objective, which naturally encourages exploration without
  hand-tuned entropy schedules.
* HER is intentionally skipped here because HandoverTask.compute_reward()
  queries live EE positions from the simulation, making it incompatible
  with vectorised goal relabelling.  The dense reward provides enough
  learning signal without HER.

Usage
─────
    # Headless training (default)
    python examples/train_dual_panda.py

    # Custom run name and timesteps
    python examples/train_dual_panda.py --run-name my_run --timesteps 500000

    # Render during training (slow — use only for debugging)
    python examples/train_dual_panda.py --render
"""

import argparse
import os
import time

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

import panda_gym  # registers PandaDualHandover-v0


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_RUN_NAME = f"sac_dual_handover_{int(time.time())}"
LOG_ROOT = "logs"
MODEL_ROOT = "models"


def make_env(render: bool = False) -> gym.Env:
    """Create and wrap one instance of the handover environment."""
    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaDualHandover-v0", render_mode=render_mode)
    env = Monitor(env)  # records episode rewards / lengths for SB3 logging
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on dual-arm handover")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                   help="Total env steps to train for (default: 1 000 000)")
    p.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME,
                   help="Unique name for this run (used for log/model paths)")
    p.add_argument("--render", action="store_true",
                   help="Render training env (debug only — very slow)")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Run evaluation callback every N steps (default: 10 000)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodes per evaluation callback run (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    log_dir = os.path.join(LOG_ROOT, args.run_name)
    model_dir = os.path.join(MODEL_ROOT, args.run_name)
    best_model_path = os.path.join(model_dir, "best_model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Environments ───────────────────────────────────────────────────────
    train_env = make_env(render=args.render)
    eval_env = make_env(render=False)

    # ── Callbacks ──────────────────────────────────────────────────────────
    # Save a checkpoint every 50 000 steps so training can be resumed.
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix="ckpt",
        verbose=1,
    )

    # Evaluate on a separate env; keep the best model found so far.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    # MultiInputPolicy handles the Dict observation space
    # (keys: "observation", "achieved_goal", "desired_goal").
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=300_000,      # keep 300k transitions in replay buffer
        learning_starts=5_000,    # warm-up period before gradient updates
        batch_size=512,
        tau=0.005,                # soft target update coefficient
        gamma=0.95,               # slightly < 1 to encourage fast success
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",            # use GPU if available, else CPU
    )

    print(f"\n{'='*60}")
    print(f"  Run  : {args.run_name}")
    print(f"  Steps: {args.timesteps:,}")
    print(f"  Logs : {log_dir}")
    print(f"  Models: {model_dir}")
    print(f"{'='*60}\n")

    # ── Training ───────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    print(f"Best model saved to : {best_model_path}/best_model.zip")
    print(f"TensorBoard logs    : {log_dir}")
    print("  -> Run:  tensorboard --logdir", log_dir)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
