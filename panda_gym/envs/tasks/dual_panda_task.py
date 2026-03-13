"""dual_panda_task.py

HandoverTask: the cube starts on the left side of the table (reachable
only by the left arm), must transit through a central handover zone, and
finally reach a target on the right side of the table (reachable only by
the right arm).

Scene objects
─────────────
  plane          – floor (z = -0.4)
  table          – wide table-top at z = 0, spanning y ∈ [-0.6, +0.6]
  object         – graspable cube (green, 4 cm side, mass 0.1 kg)
  handover_zone  – ghost flat box (blue, semi-transparent) at table centre
  target         – ghost flat box (orange, semi-transparent) on right side

Coordinate conventions (top-down view)
───────────────────────────────────────
              y
         +0.3 │  ● right arm base    [target region]
              │
         0.0  │              [handover zone]
              │
        -0.3  │  ● left arm base     [cube start]
              └──────────────────────────────> x
                -0.6        -0.1  0.0

Both arms are mounted at x = -0.6.  Their workspaces overlap in the strip
y ∈ [-0.1, +0.1] at x ≈ -0.1, which is where the handover zone lives.

GoalEnv API
───────────
  achieved_goal : cube position  (3,)
  desired_goal  : target position (3,)
  success       : distance(cube, target) < distance_threshold

Dense reward
────────────
  r = – d(cube, target)
      – α · d(active_ee, cube)     ← phase-dependent shaping

  Phase selection is based on the cube's y coordinate relative to the
  handover zone centre:
    cube_y <  handover_y + ε  →  left arm is "active"  (pick-and-carry phase)
    cube_y >= handover_y + ε  →  right arm is "active" (receive-and-place phase)

  ⚠ NOTE ON HER COMPATIBILITY:
  The shaping term queries the current EE positions from the live simulation.
  This means compute_reward() is NOT suitable for vectorised HER replay (where
  many virtual goal substitutions are evaluated on frozen sim states).
  To use HER, either:
    (a) Remove the shaping term and rely solely on –d(cube, target), or
    (b) Encode both EE positions into achieved_goal / desired_goal and
        rewrite compute_reward() to be a pure function of those arrays.
"""

from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class HandoverTask(Task):
    """Cooperative dual-arm handover task.

    Args:
        sim (PyBullet): Shared simulation instance.
        robot_left: Left arm (DualPanda or any object exposing
            ``get_left_ee_position()``).  Passed as the DualPanda wrapper so
            that reward shaping can query EE positions without importing
            DualPanda here (avoids a circular dependency).
        robot_right: Right arm accessor – same object as *robot_left* when
            using DualPanda; kept separate for clarity.
        distance_threshold (float): Success radius in metres. Defaults to 0.05.
        shaping_weight (float): Weight α for the EE-to-cube shaping term.
            Set to 0.0 to disable shaping (falls back to pure –d(cube,target)).
    """

    # Fixed scene parameters
    OBJECT_SIZE: float = 0.04          # cube half-extent × 2  (4 cm side)
    TABLE_HEIGHT: float = 0.4          # metres
    TABLE_LENGTH: float = 1.1          # x direction
    TABLE_WIDTH: float = 1.2           # y direction – wide enough for both arms
    TABLE_X_OFFSET: float = -0.3       # table centre in x

    # Handover zone: fixed at table centre (not sampled)
    HANDOVER_X: float = -0.1
    HANDOVER_Y: float = 0.0

    # Object start region (left side of table, reachable by left arm only)
    OBJ_X_CENTRE: float = -0.1
    OBJ_Y_CENTRE: float = -0.15
    OBJ_XY_RANGE: float = 0.08        # ± half-range for random start position

    # Target region (right side of table, reachable by right arm only)
    TARGET_X_CENTRE: float = -0.1
    TARGET_Y_CENTRE: float = 0.20
    TARGET_X_RANGE: float = 0.05
    TARGET_Y_RANGE: float = 0.08

    def __init__(
        self,
        sim: PyBullet,
        dual_robot,                     # DualPanda instance
        distance_threshold: float = 0.05,
        shaping_weight: float = 0.3,
    ) -> None:
        super().__init__(sim)

        # Keep a reference to the DualPanda wrapper so reward shaping can
        # query EE positions.  The task does NOT call robot methods during
        # scene creation – only during compute_reward() at run time.
        self._dual_robot = dual_robot

        self.distance_threshold = distance_threshold
        self.shaping_weight = shaping_weight
        self.object_size = self.OBJECT_SIZE

        # Handover zone centre (z = table surface + half cube height)
        self.handover_center = np.array(
            [self.HANDOVER_X, self.HANDOVER_Y, self.object_size / 2]
        )

        with self.sim.no_rendering():
            self._create_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _create_scene(self) -> None:
        """Populate the PyBullet world with all static and dynamic bodies."""
        half_obj = self.object_size / 2

        # Floor
        self.sim.create_plane(z_offset=-self.TABLE_HEIGHT)

        # Wide table (top surface at z = 0)
        self.sim.create_table(
            length=self.TABLE_LENGTH,
            width=self.TABLE_WIDTH,
            height=self.TABLE_HEIGHT,
            x_offset=self.TABLE_X_OFFSET,
            lateral_friction=0.8,
        )

        # Graspable cube – starts on the left side of the table
        self.sim.create_box(
            body_name="object",
            half_extents=np.full(3, half_obj),
            mass=0.1,
            position=np.array([self.OBJ_X_CENTRE, self.OBJ_Y_CENTRE, half_obj]),
            rgba_color=np.array([0.1, 0.8, 0.1, 1.0]),   # green
            lateral_friction=1.0,
            spinning_friction=0.005,
        )

        # Handover zone – ghost (no collision), semi-transparent blue flat slab
        # Visual only; used by reward to detect phase transition.
        self.sim.create_box(
            body_name="handover_zone",
            half_extents=np.array([0.08, 0.08, 0.002]),  # thin flat marker
            mass=0.0,
            ghost=True,
            position=np.array([self.HANDOVER_X, self.HANDOVER_Y, 0.001]),
            rgba_color=np.array([0.2, 0.4, 1.0, 0.35]),  # translucent blue
        )

        # Target zone – ghost, semi-transparent orange flat slab
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([0.06, 0.06, 0.002]),  # thin flat marker
            mass=0.0,
            ghost=True,
            position=np.array([self.TARGET_X_CENTRE, self.TARGET_Y_CENTRE, 0.001]),
            rgba_color=np.array([1.0, 0.55, 0.0, 0.45]),  # translucent orange
        )

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Randomise cube start position and target position for a new episode."""
        self.goal = self._sample_goal()
        cube_pos = self._sample_object()

        # Move target marker to sampled goal
        self.sim.set_base_pose(
            "target", self.goal, np.array([0.0, 0.0, 0.0, 1.0])
        )
        # Reset cube pose
        self.sim.set_base_pose(
            "object", cube_pos, np.array([0.0, 0.0, 0.0, 1.0])
        )

    def _sample_goal(self) -> np.ndarray:
        """Sample a target position on the right side of the table."""
        dx = self.np_random.uniform(-self.TARGET_X_RANGE, self.TARGET_X_RANGE)
        dy = self.np_random.uniform(-self.TARGET_Y_RANGE, self.TARGET_Y_RANGE)
        return np.array(
            [
                self.TARGET_X_CENTRE + dx,
                self.TARGET_Y_CENTRE + dy,
                self.object_size / 2,   # cube rests on table surface
            ]
        )

    def _sample_object(self) -> np.ndarray:
        """Sample a starting position for the cube on the left side."""
        dxy = self.np_random.uniform(-self.OBJ_XY_RANGE, self.OBJ_XY_RANGE, size=2)
        return np.array(
            [
                self.OBJ_X_CENTRE + dxy[0],
                self.OBJ_Y_CENTRE + dxy[1],
                self.object_size / 2,   # rests on table
            ]
        )

    def get_obs(self) -> np.ndarray:
        """Return the task-specific part of the observation.

        Returns:
            np.ndarray: 12-dim vector
                [cube_pos(3), cube_euler(3), cube_vel(3), cube_avel(3)]
        """
        cube_pos = self.sim.get_base_position("object")
        cube_rot = self.sim.get_base_rotation("object")          # Euler angles
        cube_vel = self.sim.get_base_velocity("object")
        cube_avel = self.sim.get_base_angular_velocity("object")
        return np.concatenate(
            [cube_pos, cube_rot, cube_vel, cube_avel]
        ).astype(np.float32)

    def get_achieved_goal(self) -> np.ndarray:
        """Return the current cube position as the achieved goal.

        Returns:
            np.ndarray: (3,) cube position in world frame.
        """
        return np.array(self.sim.get_base_position("object"), dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward and success
    # ------------------------------------------------------------------

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        """Episode succeeds when the cube reaches the target zone.

        Returns:
            np.ndarray: bool scalar.
        """
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        """Dense reward encouraging cooperative handover.

        Reward components:
          1. Primary  : –distance(cube, target)      (always active)
          2. Shaping  : –α · distance(active_ee, cube)
                        Left arm is "active" while cube has not yet crossed
                        the handover zone; right arm becomes active after.

        ⚠ The shaping term queries live simulation state and is therefore
          NOT compatible with vectorised HER goal relabelling.

        Args:
            achieved_goal: Cube position (3,).
            desired_goal:  Target position (3,).
            info:          Unused; kept for API compatibility.

        Returns:
            np.ndarray: scalar float32 reward.
        """
        cube_pos = np.asarray(achieved_goal, dtype=np.float64)
        target_pos = np.asarray(desired_goal, dtype=np.float64)

        # ── Primary objective ────────────────────────────────────────────
        d_to_target = distance(cube_pos, target_pos)

        # ── Phase-dependent EE shaping ───────────────────────────────────
        # Phase boundary: has the cube crossed the handover zone in y?
        # We use a small tolerance (half the handover zone half-extent, 0.08 m)
        # to give a smooth transition.
        cube_past_handover = cube_pos[1] >= (self.handover_center[1] + 0.05)

        if self.shaping_weight > 0.0:
            if cube_past_handover:
                # Right arm should now approach the cube and carry it to target.
                active_ee = self._dual_robot.get_right_ee_position()
            else:
                # Left arm should pick up the cube and bring it to handover zone.
                active_ee = self._dual_robot.get_left_ee_position()

            d_ee_to_cube = distance(np.asarray(active_ee, dtype=np.float64), cube_pos)
            shaping = self.shaping_weight * d_ee_to_cube
        else:
            shaping = 0.0

        reward = -(d_to_target + shaping)
        return np.float32(reward)
