from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

env = gym.make('Contra-v0')
env = JoypadSpace(env, RIGHT_ONLY)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])

done = True
env.reset()
for step in range(5000):
    actions=env.action_space.sample()
    state, reward, terminated,truncated, info = env.step(actions)
    done = terminated or truncated
    if done:
        print("over")
        break
    env.render()

env.close()