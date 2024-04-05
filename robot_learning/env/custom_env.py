import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model_path):
        super(CustomEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_string(model_path)
        self.sim = mujoco.MjSim(self.model)
        self.viewer = None

        # Define action and observation space
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.model.nu,), dtype='float32')
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.model.nq + self.model.nv,), dtype='float32')

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._is_done(obs)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
        initial_obs = self._get_obs()
        return initial_obs

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        # extract observation from the simulator
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    def _get_reward(self, obs):
        # first reward component: sum of ditance between the gripper and grasp point
        # second reward component: successful grasping
        # third reward component: safty penaly such as collision
        return 0

    def _is_done(self, obs):
        # early terminataion possible
        return False

