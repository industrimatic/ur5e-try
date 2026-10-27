import mujoco
import mujoco.viewer
import numpy as np
import time


def main(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    start_pos = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
    data.qpos[:6] = start_pos
    mujoco.mj_forward(model, data)
    data.ctrl[:6] = start_pos

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    xml_path = "./universal_robots_ur5e/scene.xml"
    main(xml_path)
