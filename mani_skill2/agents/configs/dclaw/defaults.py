from copy import deepcopy

from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class DClawDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/dclaw_description/DClaw.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_joint_names = [
            "Joint1_1",
            "Joint2_1",
            "Joint3_1",
            "Joint1_2",
            "Joint2_2",
            "Joint3_2",
            "Joint1_3",
            "Joint2_3",
            "Joint3_3",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = []
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "base_link"

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        # gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
        #     self.gripper_joint_names,
        #     -0.01,  # a trick to have force when the object is thin
        #     0.04,
        #     self.gripper_stiffness,
        #     self.gripper_damping,
        #     self.gripper_force_limit,
        # )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel,),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[0.0464982, -0.0200011, 0.0360011],
            q=[0, 0.70710678, 0, 0.70710678],
            width=128,
            height=128,
            fov=1.57,
            near=0.01,
            far=10,
            actor_uid="base_link",
        )