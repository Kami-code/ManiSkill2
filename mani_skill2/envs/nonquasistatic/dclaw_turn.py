import os.path
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.core import Pose

from mani_skill2 import format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
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
        valve.set_pose(Pose(p=[0, 0, 0.33]))
        return valve

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.valve = self._build_valve()

    def _initialize_actors(self):
        self.valve.set_qpos([0])

    def _initialize_agent(self):
        if self.robot_uid == "dclaw":
            RESET_POSE = [0, -1.57, -1.57, 0, 1.57, 1.57, 0, -1.57, 1.57]
            self.agent.reset(RESET_POSE)
            pose = Pose(p=np.array([0.1, 0.09, 0.6]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi / 2, 0))
            self.agent.robot.set_pose(pose=pose)
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_task(self):
        self.valve.set_qpos([0])
        self._target_object_qpos = np.pi
        # self.goal_pos = self.box_hole_pose.p  # goal of peg head inside the hole
        # # NOTE(jigu): The goal pose is computed based on specific geometries used in this task.
        # # Only consider one side
        # self.goal_pose = (
        #     self.box.pose * self.box_hole_offset * self.peg_head_offset.inv()
        # )
        # self.peg.set_pose(self.goal_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        # if self._obs_mode in ["state", "state_dict"]:
        #     obs.update(
        #         peg_pose=vectorize_pose(self.peg.pose),
        #         peg_half_size=self.peg_half_size,
        #         box_hole_pose=vectorize_pose(self.box_hole_pose),
        #         box_hole_radius=self.box_hole_radius,
        #     )
        return obs

    def evaluate(self, **kwargs) -> dict:
        return dict(success=False)

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        target_error = self._target_object_qpos - self.valve.get_qpos()[0]
        target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi

        #
        # obs_dict = collections.OrderedDict((
        #     ('claw_qpos', claw_state.qpos),
        #     ('claw_qvel', claw_state.qvel),
        #     ('object_x', np.cos(object_state.qpos)),
        #     ('object_y', np.sin(object_state.qpos)),
        #     ('object_qvel', object_state.qvel),
        #     ('last_action', self._get_last_action()),
        #     ('target_error', target_error),
        # ))
        # # Add hardware-specific state if present.
        # if isinstance(claw_state, DynamixelRobotState):
        #     obs_dict['claw_current'] = claw_state.current
        target_dist = np.abs(target_error)

        reward -= 5 * target_dist   # Penalty for distance away from goal.


        gripper_pos = self.tcp.get_pose().p
        obj_pos = self.valve.get_pose().p
        dist = np.linalg.norm(gripper_pos - obj_pos)
        reward -= dist
        # reaching_reward = 1 - np.tanh(5 * dist)
        # reward += reaching_reward
        # reward -= np.linalg.norm(claw_vel[np.abs(claw_vel) >= 0.5])     # Penality for high velocities.

        reward += 10 * (target_dist < 0.25)     # Reward for close proximity with goal.
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

