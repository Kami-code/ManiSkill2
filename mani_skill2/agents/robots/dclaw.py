import numpy as np
import sapien.core as sapien

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.configs.dclaw import defaults
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class DClaw(BaseAgent):
    _config: defaults.DClawDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.DClawDefaultConfig()

    def _after_init(self):
        pass
        # self.finger1_link: sapien.LinkBase = get_entity_by_name(
        #     self.robot.get_links(), "panda_leftfinger"
        # )
        # self.finger2_link: sapien.LinkBase = get_entity_by_name(
        #     self.robot.get_links(), "panda_rightfinger"
        # )

    # def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
    #     assert isinstance(actor, sapien.ActorBase), type(actor)
    #     contacts = self.scene.get_contacts()
    #
    #     limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
    #     rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)
    #
    #     # direction to open the gripper
    #     ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
    #     rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]
    #
    #     # angle between impulse and open direction
    #     langle = compute_angle_between(ldirection, limpulse)
    #     rangle = compute_angle_between(rdirection, rimpulse)
    #
    #     lflag = (
    #         np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
    #     )
    #     rflag = (
    #         np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
    #     )
    #
    #     return all([lflag, rflag])
    #
    # def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
    #     assert isinstance(actor, sapien.ActorBase), type(actor)
    #     contacts = self.scene.get_contacts()
    #
    #     limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
    #     rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)
    #
    #     return (
    #         np.linalg.norm(limpulse) >= min_impulse,
    #         np.linalg.norm(rimpulse) >= min_impulse,
    #     )


