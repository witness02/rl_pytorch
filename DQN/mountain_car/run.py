import gym

from DQN.mountain_car.brain import DQNBrain

N_FEATURES = 2
N_ACTIONS = 3
LAMBDA = 0.9
MEMORY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 0.001

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    brain = DQNBrain(N_FEATURES, N_ACTIONS, GAMMA, learning_rate=LEARNING_RATE, eps_end=EPS_END, eps_start=EPS_START,
                     eps_decay=EPS_DECAY, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

    for i_episode in range(2000):
        observation = env.reset()
        episode_total_reward = 0
        step = 0
        while True:
            step += 1
            # env.render()
            action = brain.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            position, velocity = observation_
            reward = abs(position - (-0.5))
            brain.save_transition(observation, action, reward, observation_)

            brain.learn()

            episode_total_reward += reward

            observation = observation_

            if done:
                # brain.new_episode(i_episode, step)
                print("episode %d use %d steps total reward is %d" % (i_episode, step, episode_total_reward))
                break
