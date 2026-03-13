"""eval_dual_panda.py

Evaluate a trained SAC policy on the dual-arm handover environment.

Metrics reported
────────────────
  * Success rate        – fraction of episodes where the cube reached target
  * Mean episode reward – average cumulative reward across all eval episodes
  * Mean episode length – average number of steps per episode
  * Stage stats         – fraction of episodes where the cube crossed the
                          handover zone (proxy for "left arm did its job")

Usage
─────
    # Evaluate the best model saved by train_dual_panda.py
    python examples/eval_dual_panda.py --model models/<run_name>/best_model/best_model

    # Render while evaluating
    python examples/eval_dual_panda.py --model <path> --render --episodes 10

    # Evaluate a specific checkpoint
    python examples/eval_dual_panda.py --model models/<run_name>/ckpt_500000_steps
"""

import argparse
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

import panda_gym  # registers PandaDualHandover-v0


# Handover zone y-threshold (must match HandoverTask.HANDOVER_Y + tolerance)
HANDOVER_Y_THRESHOLD = 0.05   # cube_y >= this value → left arm succeeded


def make_env(render: bool = False) -> gym.Env:
    render_mode = "human" if render else "rgb_array"
    return gym.make("PandaDualHandover-v0", render_mode=render_mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained SAC policy")
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to saved model (without .zip extension)",
    )
    p.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Render each episode visually",
    )
    p.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic policy actions (default: True)",
    )
    p.add_argument(
        "--stochastic", dest="deterministic", action="store_false",
        help="Use stochastic policy actions instead",
    )
    return p.parse_args()


def evaluate(
    model: SAC,
    env: gym.Env,
    n_episodes: int,
    deterministic: bool,
) -> dict:
    """Run n_episodes and return a dict of aggregate statistics.

    Returns:
        dict with keys:
            success_rate     float  in [0, 1]
            mean_reward      float
            std_reward       float
            mean_ep_length   float
            handover_rate    float  – fraction where cube crossed handover zone
            per_episode      list[dict]  – per-episode breakdown
    """
    episode_rewards = []
    episode_lengths = []
    successes = []
    handovers = []   # did cube cross the handover y-threshold at any point?

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        handover_achieved = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

            # Track stage: has the cube crossed the handover zone?
            # The cube position is the achieved_goal in the obs dict.
            cube_y = obs["achieved_goal"][1]
            if cube_y >= HANDOVER_Y_THRESHOLD:
                handover_achieved = True

        success = bool(info.get("is_success", False))
        successes.append(success)
        handovers.append(handover_achieved)
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        print(
            f"  Ep {ep+1:>3}/{n_episodes}  "
            f"reward={ep_reward:+7.3f}  "
            f"steps={ep_length:>3}  "
            f"handover={'YES' if handover_achieved else 'no ':>3}  "
            f"success={'YES' if success else 'no '}"
        )

    return dict(
        success_rate=float(np.mean(successes)),
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_ep_length=float(np.mean(episode_lengths)),
        handover_rate=float(np.mean(handovers)),
        per_episode=[
            {"reward": r, "length": l, "handover": h, "success": s}
            for r, l, h, s in zip(
                episode_rewards, episode_lengths, handovers, successes
            )
        ],
    )


def main() -> None:
    args = parse_args()

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = args.model
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}' (tried with and without .zip).\n"
            "Train first:  python examples/train_dual_panda.py"
        )

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, device="auto")

    # ── Environment ────────────────────────────────────────────────────────
    env = make_env(render=args.render)

    print(f"\nEvaluating over {args.episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'} policy)\n")
    print("-" * 65)

    stats = evaluate(model, env, args.episodes, args.deterministic)

    # ── Summary ────────────────────────────────────────────────────────────
    print("-" * 65)
    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Success rate    : {stats['success_rate']*100:.1f}%")
    print(f"  Handover rate   : {stats['handover_rate']*100:.1f}%  "
          "(cube crossed handover zone)")
    print(f"  Mean reward     : {stats['mean_reward']:+.3f} "
          f"± {stats['std_reward']:.3f}")
    print(f"  Mean ep length  : {stats['mean_ep_length']:.1f} steps")

    env.close()


if __name__ == "__main__":
    main()
