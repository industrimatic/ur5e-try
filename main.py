import mujoco
import mujoco.viewer
import time


def main(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.ctrl[:6] = [-1.57, -1.35, 2.65, -1.3, 1.55, 0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    xml_path = "./universal_robots_ur5e/scene.xml"
    main(xml_path)
