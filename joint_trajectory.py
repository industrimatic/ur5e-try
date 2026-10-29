import mujoco
import mujoco.viewer
import ikpy.chain
import numpy as np
import transforms3d as tf
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


def main(xml_path: str, urdf_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file(
        urdf_path, active_links_mask=[False, False] + [True] * 6 + [False]
    )

    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
    data.qpos[:6] = start_joints

    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    joint_angles = my_chain.inverse_kinematics(
        ee_pos, ee_orientation, "all", initial_position=ref_pos
    )
    end_joints = joint_angles[2:-1]

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])
            data.ctrl[:6] = waypoint

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main(
        "universal_robots_ur5e/scene_changed.xml",
        "universal_robots_ur5e/ur5e_orig.urdf",
    )
