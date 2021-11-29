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

env = gym.make("CartPole-v1")

D = deque(maxlen=10000)

inputs = layers.Input(shape=((4,)))
hidden_layer_1 = layers.Dense(256, activation='relu')(inputs)
hidden_layer_2 = layers.Dense(256, activation='relu')(hidden_layer_1)
outputs = layers.Dense(2, activation='linear')(hidden_layer_2)

model = keras.Model(inputs= inputs, outputs= outputs)
target = keras.models.clone_model(model)
model.compile(loss='mse', optimizer=Adam())

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
        choose_action = random.randrange(1, 101)
        if choose_action < epsilon*100:
            action = random.randrange(0, 2)
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
                    y[eval[1]] = eval[2] + ganma*np.argmax(target.predict(np.expand_dims(eval[3], 0)))

                state_batch.append(eval[0])
                q_value_batch.append(y)
            model.fit(np.array(state_batch), np.array(q_value_batch), epochs=1, verbose=0)
            reset_counter += 1
            if reset_counter % 10 == 0:
                target = keras.models.clone_model(model)
        if done == True:
            save_counter += 1
            if save_counter % 10 == 0:
                model.save('my_model')
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            print(total_reward)
            if total_reward > final_reward:
                final_reward = total_reward
            break

model.save('my_model')

                
