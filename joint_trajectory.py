import mujoco
import mujoco.viewer
import numpy as np
from typing import Iterator


class JointSpaceTrajectory:
    def __init__(
        self, start_joints: np.ndarray, end_joints: np.ndarray, steps: int
    ) -> None:
        self.start_joints = start_joints
        self.end_joints = end_joints
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.way_points = self.start_joints

    def _generate_trajectory(self) -> Iterator[np.ndarray]:
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i

        yield self.end_joints

    def get_next_waypoint(self, pos_now):
        if np.allclose(pos_now, self.way_points, atol=0.02):
            try:
                self.way_points = next(self.trajectory)
                return self.way_points
            except StopIteration:
                pass
        return self.way_points
