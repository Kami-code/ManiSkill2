import gymnasium as gym
import mani_skill2.envs



if __name__ == "__main__":
    env = gym.make("DClawTurn-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="human")
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    obs, reset_info = env.reset(seed=0) # reset with a seed for randomness
    terminated, truncated = False, False
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward)
        env.render()  # a display is required to render
    env.close()