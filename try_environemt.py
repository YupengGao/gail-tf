# from gym_soccer.envs import SoccerEnv
from gailtf.common import convert_log2_tensor
import gym_soccer
import gym
env = gym.make('SoccerEmptyGoal-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)

        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break