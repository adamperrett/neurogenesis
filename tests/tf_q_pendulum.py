
#encoding: utf-8
# https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
##
## cartpole.py
## Gaetan JUVIN 06/24/2017
##

import gym
import random
import os
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lr = 0.001
gamma = 0.95
hidden_size = 24
batch_size = 4
extra_layers = 0

test_label = 'bp q learning lr-{} gamma-{} hidden_size-{}x{} batch_size-{}'.format(
    lr, gamma, hidden_size, extra_layers, batch_size)


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = lr
        self.gamma              = gamma
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(hidden_size, input_dim=self.state_size, activation='relu'))
        for l in range(extra_layers):
            model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = batch_size
        self.episodes          = 2000
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)


    def run(self):
        # try:
        balance_lengths = []
        for index_episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            balance_count = 0
            while not done:
#                    self.env.render()

                action = self.agent.act(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                balance_count += 1
            print("Episode {}# Score: {}".format(index_episode, balance_count))
            balance_lengths.append(balance_count)
            average_reward = np.average(balance_lengths[-100:])
            if index_episode % 10 == 0:
                print(test_label)
                template = "average reward: {:.2f} at episode {} of trial {}"
                print(template.format(average_reward, index_episode, repeat))
            if len(balance_lengths) > 100 and average_reward > 475:
                print(test_label)
                print("Solved at episode {}!".format(index_episode))
                print("Average reward:", average_reward)
                # learning_history.append(balance_lengths)
                break
            self.agent.replay(self.sample_batch_size)
        # finally:
        #     self.agent.save_model()
        if index_episode >= self.episodes-1:
            solved = self.episodes * 2
        else:
            solved = index_episode
        return balance_lengths, solved

if __name__ == "__main__":
    repeats = 100
    learning_history = []
    solve_history = []
    for repeat in range(repeats):
        cartpole = CartPole()
        bl, sl = cartpole.run()
        learning_history.append(bl)
        solve_history.append(sl)
        print("Solves so far:", solve_history, "\nThe average solve length is", np.average(solve_history))
        extended_data = []
        for lh in learning_history:
            extended_data.append(lh)
            while len(extended_data[-1]) < 2000:
                extended_data[-1].append(500.)
        plt.figure()
        for ed in extended_data:
            plt.plot([i for i in range(len(ed))], ed, 'r')
        average_balance = [np.average([extended_data[i][ts] for i in range(len(extended_data))]) for ts in range(2000)]
        plt.plot([i for i in range(len(average_balance))], average_balance, 'b')
        plt.plot([0, 2000], [475, 475], 'g')
        figure = plt.gcf()
        figure.set_size_inches(16, 9)
        plt.tight_layout(rect=[0, 0.3, 1, 0.95])
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200)
        plt.close()

    np.save("./data/{}".format(test_label), [learning_history, solve_history])

    print("done")

