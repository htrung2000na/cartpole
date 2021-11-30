import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque 
from tensorflow.keras.optimizers import Adam

max_episode = 500
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = epsilon_min/epsilon
epsilon_decay = epsilon_decay**(1. / float(max_episode))
batch_size = 10
ganma = 0.99
reset_counter = 0
save_counter = 0
load_model = True

env = gym.make("CartPole-v1")

inputs = layers.Input(shape=((4,)))
hidden_layer_1 = layers.Dense(256, activation='relu')(inputs)
hidden_layer_2 = layers.Dense(256, activation='relu')(hidden_layer_1)
outputs = layers.Dense(2, activation='linear')(hidden_layer_2)

if load_model == True:
    model = keras.models.load_model('my_model')
    D = np.load('replay_memory.npy', allow_pickle=True)
    D = deque(D, maxlen=10000)
else:
    model = keras.Model(inputs= inputs, outputs= outputs)
    model.compile(loss='mse', optimizer=Adam())
    D = deque(maxlen=10000)
target = keras.models.clone_model(model)

final_reward = 0
current_episode = 0
while final_reward < 100:
    state = env.reset()
    current_episode += 1
    if current_episode >= max_episode:
        break
    total_reward = 0
    for i in range(1000):
        # env.render()
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(state, 0)))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        D.append((state, action, reward, next_state, done))
        if len(D) >= batch_size:
            state_batch = []
            q_value_batch = []
            mini_batch = random.sample(D, k = batch_size)
            for eval in mini_batch:
                y = np.zeros(2)
                if eval[4] == True:
                    y[eval[1]] = eval[2]
                else:
                    y[eval[1]] = eval[2] + ganma*np.amax(target.predict(np.expand_dims(eval[3], 0)))

                state_batch.append(eval[0])
                q_value_batch.append(y)
            model.fit(np.array(state_batch), np.array(q_value_batch), batch_size=batch_size, epochs=1, verbose=0)
            reset_counter += 1
            if reset_counter % 10 == 0:
                target = keras.models.clone_model(model)
        if done == True:
            save_counter += 1
            if save_counter % 10 == 0:
                model.save('my_model')
                np.save('replay_memory.npy', D)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            print("running reward: {:.2f} at episode {}".format(total_reward, current_episode))
            if total_reward > final_reward:
                final_reward = total_reward
            break

model.save('my_model')

                
