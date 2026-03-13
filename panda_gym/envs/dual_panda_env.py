"""dual_panda_env.py

DualPandaHandoverEnv – Gymnasium environment for a dual-arm cube handover.

Two Franka Panda robots sit side-by-side on a wide table.  The left arm
must pick up a cube and bring it to a central handover zone; the right arm
then takes the cube and places it on the target region.

Both arms are controlled jointly by a single policy (single-agent RL).

Quick-start
───────────
    import panda_gym                         # optional if using direct import
    from panda_gym.envs.dual_panda_env import DualPandaHandoverEnv

    env = DualPandaHandoverEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            obs, info = env.reset()
    env.close()

Spaces (default: control_type="ee", block_gripper=False)
─────────────────────────────────────────────────────────
    action_space     Box(8,)   [-1, 1]
                     [ dx_L, dy_L, dz_L, grip_L,
                       dx_R, dy_R, dz_R, grip_R ]

    observation_space  Dict:
      "observation"    Box(26,)
                       [ left_ee_pos(3), left_ee_vel(3), left_grip(1),
                         right_ee_pos(3), right_ee_vel(3), right_grip(1),
                         cube_pos(3), cube_euler(3), cube_vel(3), cube_avel(3) ]
      "achieved_goal"  Box(3,)   cube position
      "desired_goal"   Box(3,)   target position

    reward    dense float (see HandoverTask.compute_reward for formula)
    success   info["is_success"] = True when cube reaches target within
              distance_threshold metres
"""

from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.dual_panda_robot import DualPanda
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.dual_panda_task import HandoverTask
from panda_gym.pybullet import PyBullet


class DualPandaHandoverEnv(RobotTaskEnv):
    """Dual-arm handover environment built on top of RobotTaskEnv.

    Uses *two* Panda instances loaded from the same URDF under different
    body names ("panda_left", "panda_right"), wrapped by DualPanda to
    present a single combined action / observation interface that
    RobotTaskEnv expects.

    Args:
        render_mode (str): "rgb_array" (default) or "human".
        control_type (str): "ee" (default) for end-effector Cartesian control
            or "joints" for direct joint-angle control.
        distance_threshold (float): Goal success radius in metres.
            Defaults to 0.05.
        shaping_weight (float): Weight of the EE-to-cube shaping term in
            the dense reward.  0.0 disables shaping.  Defaults to 0.3.
        renderer (str): PyBullet renderer.  "Tiny" (fast, headless) or
            "OpenGL" (for visualisation).  Defaults to "Tiny".
        render_width (int): Pixel width of rendered images.  Defaults to 720.
        render_height (int): Pixel height of rendered images.  Defaults to 480.
        render_target_position (np.ndarray, optional): Camera look-at point.
            Defaults to the table centre [−0.1, 0.0, 0.0].
        render_distance (float): Camera distance from target.  Defaults to 1.8.
        render_yaw (float): Camera yaw in degrees.  Defaults to 90 (side view).
        render_pitch (float): Camera pitch in degrees.  Defaults to −30.
        render_roll (float): Camera roll in degrees.  Defaults to 0.
    """

    # Base positions of the two arms.
    # Both are at x = -0.6 (same depth as single-arm envs), separated along y.
    # Workspace overlap occurs in the strip y ∈ [-0.1, +0.1] at x ≈ -0.1,
    # which is exactly where the handover zone is placed.
    LEFT_BASE: np.ndarray = np.array([-0.6, -0.3, 0.0])
    RIGHT_BASE: np.ndarray = np.array([-0.6, +0.3, 0.0])

    def __init__(
        self,
        render_mode: str = "rgb_array",
        control_type: str = "ee",
        distance_threshold: float = 0.05,
        shaping_weight: float = 0.3,
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.8,
        render_yaw: float = 90.0,
        render_pitch: float = -30.0,
        render_roll: float = 0.0,
    ) -> None:

        # ── 1. Shared physics simulation ─────────────────────────────────
        sim = PyBullet(render_mode=render_mode, renderer=renderer)

        # ── 2. Individual arm instances ──────────────────────────────────
        # Each arm gets a unique body_name so sim._bodies_idx has two separate
        # entries and IK / joint queries are routed to the correct body.
        robot_left = Panda(
            sim,
            block_gripper=False,
            base_position=self.LEFT_BASE.copy(),
            control_type=control_type,
            body_name="panda_left",
        )
        robot_right = Panda(
            sim,
            block_gripper=False,
            base_position=self.RIGHT_BASE.copy(),
            control_type=control_type,
            body_name="panda_right",
        )

        # ── 3. Dual-arm wrapper (combines action / obs spaces) ───────────
        dual_robot = DualPanda(sim, robot_left, robot_right)

        # ── 4. Task (creates scene objects; receives robot refs for reward) ─
        task = HandoverTask(
            sim,
            dual_robot=dual_robot,
            distance_threshold=distance_threshold,
            shaping_weight=shaping_weight,
        )

        # ── 5. Compose into Gymnasium env ────────────────────────────────
        # RobotTaskEnv calls reset() here to infer observation / action shapes,
        # so everything above must be fully initialised before super().__init__.
        render_target_position = (
            render_target_position
            if render_target_position is not None
            else np.array([-0.1, 0.0, 0.0])  # centre of the wide table
        )
        super().__init__(
            robot=dual_robot,
            task=task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
