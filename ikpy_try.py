import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d as tf
import numpy as np


def main():
    chain = ikpy.chain.Chain.from_urdf_file("./universal_robots_ur5e/ur5e_orig.urdf")
    target_pos = np.array([-0.31, 0.5, 0.1])
    target_euler = np.array([np.pi, 0, np.pi / 2])
    target_matrix = tf.euler.euler2mat(*target_euler)
    ref_pos = [0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]

    fig, ax = plot_utils.init_3d_figure()
    chain.plot(
        chain.inverse_kinematics(
            target_pos, target_matrix, "all", initial_position=ref_pos
        ),
        ax,
    )
    plt.show()


if __name__ == "__main__":
    main()
