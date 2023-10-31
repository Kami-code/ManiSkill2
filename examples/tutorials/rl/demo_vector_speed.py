import argparse
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill2.utils.visualization.misc import observations_to_images, tile_images
from mani_skill2.vector import VecEnv, make
import torch.nn as nn
import torch
from icecream import ic

class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f'PointNetSmall')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # pc = x[0].cpu().detach().numpy()
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x

class PointNetMedium(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetMedium, self).__init__()

        print(f'PointNetMedium')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLarge(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLarge, self).__init__()

        print(f'PointNetLarge')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PickCube-v0",
        help="The environment to this demo on",
    )
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        default="pointcloud",
        help="The observation mode to use",
    )
    parser.add_argument(
        "-c", "--control-mode", type=str, help="The control mode to use"
    )
    parser.add_argument("--reward-mode", type=str, help="The reward mode to use")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Whether to visualize the environments"
    )
    parser.add_argument(
        "--n-ep",
        type=int,
        default=5,
        help="Number of episodes to run per parallel environment",
    )
    parser.add_argument(
        "--l-ep", type=int, default=200, help="Max number of timesteps per episode"
    )
    parser.add_argument(
        "--vecenv-type",
        type=str,
        default="ms2",
        help="Type of VecEnv to use. Can be ms2 or gym",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main(args):
    np.set_printoptions(suppress=True, precision=3)

    verbose = not args.quiet
    n_ep = args.n_ep
    l_ep = args.l_ep

    if args.vecenv_type == "ms2":
        env: VecEnv = make(
            args.env_id,
            args.n_envs,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            **args.env_kwargs,
        )
    elif args.vecenv_type == "gym":
        env = gym.make_vec(
            args.env_id,
            args.n_envs,
            vectorization_mode="async",
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            vector_kwargs=dict(context="forkserver"),
        )
    else:
        raise ValueError(f"{args.vecenv_type} is invalid. Must be ms2 or gym")
    if verbose:
        print(f"Environment {args.env_id} - {args.n_envs} parallel envs")
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)

    np.random.seed(2022)

    samples_so_far = 0
    total_samples = n_ep * l_ep * args.n_envs
    tic = time.time()
    pn = PointNetLarge().cuda()
    pbar = tqdm(range(n_ep))
    for i in pbar:
        # NOTE(jigu): reset is a costly operation
        obs, _ = env.reset()

        for t in range(l_ep):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            pn.forward(obs['pointcloud']['xyzw'][:, :, :3])
            samples_so_far += args.n_envs

        fps = samples_so_far / (time.time() - tic)
        pbar.set_postfix(dict(FPS=f"{fps:0.2f}"))
    toc = time.time()
    if verbose:
        print(f"FPS {total_samples / (toc - tic):0.2f}")
    env.close()


if __name__ == "__main__":
    main(parse_args())
