#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ManiSkill2 
@File    ：robot_size_loader.py
@Author  ：Chen Bao
@Date    ：2023/5/22 下午3:19 
'''
import sapien.core as sapien
from sapien.utils import Viewer

if __name__ == '__main__':
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: sapien.Articulation = loader.load("../mani_skill2/assets/descriptions/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: sapien.Articulation = loader.load("../mani_skill2/assets/descriptions/panda_v2.urdf")
    robot.set_root_pose(sapien.Pose([1, 0, 0], [1, 0, 0, 0]))
    while not viewer.closed:
        scene.update_render()
        viewer.render()