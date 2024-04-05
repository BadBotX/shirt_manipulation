import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import time
import mujoco.viewer

# Load your existing XML file
xml_path = '/home/lxu/LizhouXu/Repo/shirt_manipulation/models/world/dual_arm_table_arena.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

# Find the worldbody element (assuming you're adding the sphere to the worldbody)
worldbody = root.find('.//worldbody')

# Generate random noise for the sphere's position
noise = np.random.normal(0, 0.05, 3)  # Mean 0, standard deviation 0.05, for each XYZ coordinate

original_position = np.array([0, 0, 5])
noisy_position = original_position + noise

sphere = ET.Element('geom')
sphere.set('type', 'sphere')
sphere.set('size', '0.8') 
sphere.set('pos', f"{noisy_position[0]} {noisy_position[1]} {noisy_position[2]}")

worldbody.append(sphere)

tree.write('modified_environment.xml')

xml_string = ET.tostring(root, encoding='unicode')
m = mujoco.MjModel.from_xml_string(xml_string)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 6000:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)