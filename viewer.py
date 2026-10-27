import mujoco
import mujoco.viewer

xml_path = "./universal_robots_ur5e/ur5e.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
