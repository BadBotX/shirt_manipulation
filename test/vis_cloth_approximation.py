import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import time
import mujoco.viewer

current_path = Path(__file__).parent
main_env_path = current_path.parent / 'models'/'world' / 'table_arena.xml'
main_tree = ET.parse(main_env_path)
main_root = main_tree.getroot()


model_to_add_path = current_path.parent / 'models'/'cloth_approximation' / 'poloshirt.xml'
model_tree = ET.parse(model_to_add_path)
model_root = model_tree.getroot()

# initialization noise
mean = 0
std_dev = 0.1
x_initialization_noise = np.random.normal(mean, std_dev)
y_initialization_noise = np.random.normal(mean, std_dev)
floats = [x_initialization_noise, y_initialization_noise, 0.9]
cloth_initial_position = ' '.join(str(f) for f in floats)
rotation_initialization_noise = np.random.normal(mean, 90)
floats = [0, 0, rotation_initialization_noise]
cloth_initial_rotation = ' '.join(str(f) for f in floats)

model_worldbody = model_root.find('worldbody')
if model_worldbody is not None:
    # Find the worldbody in the main environment to append to
    main_worldbody = main_root.find('.//worldbody')
    if main_worldbody is not None:
        for elem in list(model_worldbody):
            top_level_body_of_model = list(model_worldbody)[0]
            top_level_body_of_model.set('pos', cloth_initial_position)
            top_level_body_of_model.set('euler', cloth_initial_rotation )
            #top_level_body_of_model.set('pos', '0 0.55 0.9')
            main_worldbody.append(top_level_body_of_model)

modified_xml_string = ET.tostring(main_root, encoding='unicode')
# with open('modified_environment.xml', 'w', encoding='utf-8') as file:
#     file.write(modified_xml_string)

m = mujoco.MjModel.from_xml_string(modified_xml_string)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running() and time.time() - start < 6000:
    step_start = time.time()
    mujoco.mj_step(m, d)
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
    viewer.sync()
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)