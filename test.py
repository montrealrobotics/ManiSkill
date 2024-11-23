import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import camera_observations_to_images


env: BaseEnv = gym.make(
  "PutCarrotOnPlateInSceneSep-v1",
  obs_mode="state_dict",
  render_mode="human",
  num_envs=1, # if num_envs > 1, GPU simulation backend is used.
  sim_backend="cpu",
  control_mode="pd_joint_delta_pos",
  reward_mode="dense",
  robot="panda_robotiq",
)
obs, _ = env.reset()
print(obs)
# images = camera_observations_to_images(obs["sensor_data"]["3rd_view_camera"])
# rgb = np.concatenate(images["rgb"].cpu().numpy(), axis=1)
# segment = np.concatenate(images["segmentation"].cpu().numpy(), axis=1)

# combined = np.concatenate([rgb, segment], axis=0)
# plt.imshow(combined)
# plt.show()

while True:
    action = env.action_space.sample() # replace this with your policy inference
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
