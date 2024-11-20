import argparse

import gymnasium as gym
import numpy as np
import sapien
import time
from PIL import Image

from octo.data.dataset import make_single_dataset
from octo.data.oxe.oxe_standardization_transforms import droid_dataset_transform
from octo.utils.spec import ModuleSpec

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-r", "--robot", type=str, default=None, help="The robot agent you want to place in the environment")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run.")
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("-p", "--pause", action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is no seed",
    )
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if args.robot:
        env_kwargs["robot"] = args.robot
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def as_gif(images, path="/home/artur/Downloads/temp.gif"):
  # Render the images as the gif (15Hz control frequency):
  print("saving video")
  images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
  gif_bytes = open(path,"rb").read()
  return gif_bytes


def main(args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if args.seed is not None:
        np.random.seed(args.seed)
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        parallel_in_single_scene=parallel_in_single_scene,
        **args.env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=env._max_episode_steps)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # ds = tfds.load("droid", data_dir="/project/", split="train")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="droid_100",
            data_dir="/home/artur/",
            # could add wrist camera here too
            image_obs_keys={"primary": "exterior_image_1_left"},
            proprio_obs_key="proprio",
            language_key="language_instruction",
            standardize_fn=ModuleSpec.create(
                droid_dataset_transform
            ),
            skip_norm=True,
        ),
        # matching octo base model
        traj_transform_kwargs=dict(
            # goal_relabeling_strategy="uniform",
            window_size=1,
            action_horizon=1,
            # subsample_length=100,
        ),
        # frame_transform_kwargs=dict(
        #     resize_size={"primary": (256, 256)},
        # ),
        train=False,
    )
    episodes = dataset.iterator()

    images = []
    for episode in episodes:
        if episode['action'].shape[0] < 50:
            continue
        for i in range(episode['action'].shape[0]):
            image = episode["observation"]["image_primary"]
            images.append(Image.fromarray(image[i,0,...]))
        as_gif(images)
        for i in range(episode['action'].shape[0]):
            # action = env.action_space.sample()
            action = episode['action'][i,0,0]
            # action = episode["observation"]["proprio"][i,0]
            # action = np.append(action, [0])
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if verbose:
                pass
                # print("reward", reward)
                # print("terminated", terminated)
                # print("truncated", truncated)
                # print("info", info)
            if args.render_mode is not None:
                env.render()
            if args.render_mode is None or args.render_mode != "human":
                if (terminated | truncated).any():
                    break
            time.sleep(1)
        break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    main(parse_args())
