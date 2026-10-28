import ikpy.chain
import transforms3d as tf
import numpy as np
import mujoco
import mujoco.viewer


def main():

    model = mujoco.MjModel.from_xml_path("./universal_robots_ur5e/scene_changed.xml")
    data = mujoco.MjData(model)
    chain = ikpy.chain.Chain.from_urdf_file(
        "./universal_robots_ur5e/ur5e_orig.urdf",
        active_links_mask=[False, False] + [True] * 6 + [False],
    )

    target_pos = np.array([-0.13, 0.5, 0.1])
    target_euler = np.array([np.pi, 0, np.pi / 2])
    target_matrix = tf.euler.euler2mat(*target_euler)
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]

    joint_angles = chain.inverse_kinematics(
        target_pos, target_matrix, "all", initial_position=ref_pos
    )
    control = joint_angles[2:-1]
    data.ctrl[:] = control

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running:

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
