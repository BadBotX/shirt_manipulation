import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
from PIL import Image
from datetime import datetime
# Load and parse the main environment XML
current_path = Path(__file__).parent
main_env_path = current_path.parent.parent / 'models'/'world' / 'table_arena.xml'
main_tree = ET.parse(main_env_path)
main_root = main_tree.getroot()

# Load and parse the XML of the model 
model_to_add_path = current_path.parent.parent / 'models'/'cloth_approximation' / 'poloshirt.xml'
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

renderer = mujoco.Renderer(m, 480, 640)

i = 0
num_steps = 10
cam1_imgs = []
cam2_imgs = []

while i < num_steps:
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera="birdview")
    img_array = renderer.render()
    if i == 8:
      timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      filename = f'image_{timestamp}.jpg'
      img = Image.fromarray(img_array)
      img.save(filename )
    i = i+1
        
