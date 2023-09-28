import os.path
from collections import OrderedDict
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2 import format_path
from mani_skill2.utils.registration import register_env

from mani_skill2.utils.sapien_utils import hex2rgba, look_at, vectorize_pose
from transforms3d.euler import euler2quat
from .base_env import NonQuasiStaticEnv


@register_env("DClawTurn-v0", max_episode_steps=200)
class DClawTurnEnv(NonQuasiStaticEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/non_quasi_static/robel/dclaw_3x.urdf"

    def __init__(self, *args, robot="dclaw", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_uid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.asset_root = Path(format_path(self.DEFAULT_ASSET_ROOT))
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        if options.get("reconfigure") is None:
            options["reconfigure"] = True
        return super().reset(seed, options)

    def _build_valve(self):
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        valve: sapien.Articulation = loader.load(str(self.asset_root))
        for joint in valve.get_joints():
            joint.set_friction(0.02)

        valve.set_qpos([0])
        valve.set_pose(Pose(p=[0, 0, 0.0], q=euler2quat(0, 0, np.pi)))
        return valve

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.valve = self._build_valve()

    def _initialize_actors(self):
        self.valve.set_qpos([0])

    def _initialize_agent(self):
        if self.robot_uid == "dclaw":
            # self.RESET_POSE = np.array([0, -1.57, -1.57, 0, 1.57, 1.57, 0, -1.57, 1.57])
            self.RESET_POSE = np.zeros((9))
            self.agent.reset(self.RESET_POSE)
            pose = Pose(p=np.array([0.135, 0.09, 0.3]), q=euler2quat(np.pi / 2, np.pi / 2, 0))
            self.agent.robot.set_pose(pose=pose)
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_task(self):
        self.valve.set_qpos([0])
        self._target_object_qpos = np.pi

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict()
        if self._obs_mode in ["state", "state_dict"]:
            target_error = self._target_object_qpos - self.valve.get_qpos()[0]
            target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi
            target_dist = np.abs(target_error)
            obs.update(
                claw_qpos=self.agent.robot.get_qpos(),
                claw_qvel=self.agent.robot.get_qvel(),
                object_x=np.cos(self.valve.get_qpos()[0]),
                object_y=np.sin(self.valve.get_qpos()[0]),
                object_qvel=self.valve.get_qvel()[0],
                target_error=target_dist
            )
        return obs

    def evaluate(self, **kwargs) -> dict:
        return dict(success=False)

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        claw_qpos = self.agent.robot.get_qpos()
        claw_qvel = self.agent.robot.get_qvel()
        target_error = self._target_object_qpos - self.valve.get_qpos()[0]
        target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi
        target_dist = np.abs(target_error)

        reward -= 5 * target_dist  # Penalty for distance away from goal.
        reward -= np.linalg.norm(claw_qpos - self.RESET_POSE)  # Penalty for difference with nomimal pose.
        reward -= np.linalg.norm(claw_qvel[np.abs(claw_qvel) > 0.5])
        reward += 10 * (target_dist < 0.25)  # Reward for close proximity with goal.
        reward += 50 * (target_dist < 0.10)
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 25.0

    def _register_cameras(self):
        cam_cfg = super()._register_cameras()
        cam_cfg.pose = look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return cam_cfg

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return cam_cfg

    def set_state(self, state):
        super().set_state(state)
        # NOTE(xuanlin): This way is specific to how we compute goals.
        # The general way is to handle variables explicitly
        self._initialize_task()
