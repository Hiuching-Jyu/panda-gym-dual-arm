"""dual_panda_robot.py

A duck-typed wrapper around two Panda instances that presents a single
concatenated action / observation interface compatible with RobotTaskEnv.

No new URDF is needed: both arms load the same franka_panda/panda.urdf
under different body names ("panda_left" / "panda_right") so they are
tracked as separate bodies in sim._bodies_idx.

Action space layout  (when control_type="ee", block_gripper=False):
    [ dx_L, dy_L, dz_L, grip_L,  dx_R, dy_R, dz_R, grip_R ]   (8-dim)

Observation layout:
    [ ee_pos_L(3), ee_vel_L(3), grip_L(1),
      ee_pos_R(3), ee_vel_R(3), grip_R(1) ]                     (14-dim)
"""

from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class DualPanda:
    """Two Franka Panda arms sharing one PyBullet simulation instance.

    This class intentionally does **not** subclass PyBulletRobot because
    PyBulletRobot represents a single body.  Instead it implements the same
    duck-typed interface that RobotTaskEnv expects:

        .sim            – shared PyBullet instance
        .action_space   – combined gym.spaces.Box
        .set_action()   – splits the vector and dispatches to each arm
        .get_obs()      – concatenates both arms' observations
        .reset()        – resets both arms to neutral pose

    Args:
        sim (PyBullet): Shared simulation instance (must also be passed to
            the task so the assertion in RobotTaskEnv passes).
        robot_left (Panda): Left arm, already loaded into *sim*.
        robot_right (Panda): Right arm, already loaded into *sim*.
    """

    def __init__(
        self,
        sim: PyBullet,
        robot_left: Panda,
        robot_right: Panda,
    ) -> None:
        self.sim = sim
        self.robot_left = robot_left
        self.robot_right = robot_right

        # Build combined action space by concatenating the two individual spaces.
        # Both arms are created with the same control_type / block_gripper flags,
        # so the shapes will match; but we keep it generic for flexibility.
        self.action_space = spaces.Box(
            low=np.concatenate(
                [robot_left.action_space.low, robot_right.action_space.low]
            ),
            high=np.concatenate(
                [robot_left.action_space.high, robot_right.action_space.high]
            ),
            dtype=np.float32,
        )

        # Cache split index so set_action() doesn't recompute it every step.
        self._n_left = robot_left.action_space.shape[0]

    # ------------------------------------------------------------------
    # Interface expected by RobotTaskEnv
    # ------------------------------------------------------------------

    def set_action(self, action: np.ndarray) -> None:
        """Split the combined action vector and forward to each arm.

        Args:
            action (np.ndarray): Concatenated action of shape
                (n_left + n_right,).  Must be clipped to action_space bounds
                before calling; each arm does its own internal clipping as well.
        """
        self.robot_left.set_action(action[: self._n_left])
        self.robot_right.set_action(action[self._n_left :])

    def get_obs(self) -> np.ndarray:
        """Return concatenated observations of both arms.

        Returns:
            np.ndarray: Shape (n_obs_left + n_obs_right,), dtype float32.
                Layout: [left_ee_pos(3), left_ee_vel(3), left_grip(1),
                         right_ee_pos(3), right_ee_vel(3), right_grip(1)]
        """
        return np.concatenate(
            [self.robot_left.get_obs(), self.robot_right.get_obs()]
        ).astype(np.float32)

    def reset(self) -> None:
        """Reset both arms to their neutral (home) joint configuration."""
        self.robot_left.reset()
        self.robot_right.reset()

    # ------------------------------------------------------------------
    # Convenience accessors (used by HandoverTask for reward shaping)
    # ------------------------------------------------------------------

    def get_left_ee_position(self) -> np.ndarray:
        """Returns the left arm end-effector position as (x, y, z)."""
        return np.array(self.robot_left.get_ee_position())

    def get_right_ee_position(self) -> np.ndarray:
        """Returns the right arm end-effector position as (x, y, z)."""
        return np.array(self.robot_right.get_ee_position())
