import gym
import numpy as np
from PG.mountain_car.agent import PolicyGradient

env = gym.make('MountainCar-v0')
env = env.unwrapped

agent = PolicyGradient(n_features=env.observation_space.shape[0],
                       n_actions=env.action_space.n,
                       lr=0.02,
                       discount=0.99
                       )

is_render = False

for i in range(200000):
    o = env.reset()
    step = 0
    if i > 50:
        is_render = True
    while True:
        step += 1
        # if is_render:
        #     env.render()
        action = agent.choose_action(o)
        _o, reward, done, _ = env.step(action)

        agent.save_transition(o, action, reward)

        o = _o

        if done:
            norm_reward = agent.learn()
            print("episode %d learn done, step is %d total reward is %f" % (i, step, np.sum(norm_reward)))
            break
