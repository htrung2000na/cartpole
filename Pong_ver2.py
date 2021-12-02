import gym
import random
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque 
from skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('PongDeterministic-v4')

no_of_actions = 3
max_episodes = 100000
obs_episodes = 1000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = epsilon_min/epsilon
epsilon_decay = epsilon_decay**(1.0 / float(max_episodes))
batch_size = 32
gamma = 0.99
replay_memory = 8000

def pre_processing(state):
    state = rgb2gray(state)
    state = resize(state, (84, 84), mode='constant')
    state = np.uint8(state)
    return state

def create_model():
    inputs = layers.Input((84, 84, 4))
    hidden_layer_1 = layers.Conv2D(16, 8, 4, activation='relu')(inputs)
    hidden_layer_2 = layers.Conv2D(32, 4, 2, activation='relu')(hidden_layer_1)
    hidden_layer_3 = layers.Flatten()(hidden_layer_2)
    hidden_layer_4 = layers.Dense(256, activation='relu')(hidden_layer_3)
    outputs = layers.Dense(no_of_actions, activation='linear')(hidden_layer_4)

    model = keras.Model(inputs= inputs, outputs= outputs)
    model.compile(loss=tf.keras.losses.Huber(), optimizer='Adam')
    return model

def create_target_model():
    inputs = layers.Input((84, 84, 4))
    hidden_layer_1 = layers.Conv2D(16, 8, 4, activation='relu')(inputs)
    hidden_layer_2 = layers.Conv2D(32, 4, 2, activation='relu')(hidden_layer_1)
    hidden_layer_3 = layers.Flatten()(hidden_layer_2)
    hidden_layer_4 = layers.Dense(256, activation='relu')(hidden_layer_3)
    outputs = layers.Dense(no_of_actions, activation='linear')(hidden_layer_4)

    model = keras.Model(inputs= inputs, outputs= outputs)
    model.compile(loss=tf.keras.losses.Huber(), optimizer='Adam')
    return model

def get_action(model, history):
    if np.random.rand() <= epsilon:
        return random.randrange(no_of_actions)
    else:
        return np.argmax(model.predict(np.expand_dims(history, 0))[0])

def train_memory_batch(memory, model, target_model):
    mini_batch = random.sample(memory, k=batch_size)
    history_batch = []
    q_values_batch = []
    for eval in mini_batch:
        y = model.predict(np.expand_dims(eval[0], 0))
        if eval[4] == True:
            y[0][eval[1]] = eval[2]
        else:
            y[0][eval[1]] = eval[2] + gamma*np.amax(target_model.predict(np.expand_dims(eval[3], 0))[0])
        history_batch.append(eval[0])
        q_values_batch.append(y[0])
    model.fit(np.array(history_batch), np.array(q_values_batch), batch_size=batch_size, epochs=1, verbose=0)

def train(epsilon):
    memory = deque(maxlen=replay_memory)
    model = create_model()
    target_model = create_target_model()
    reset_counter = 0
    for episode_number in range(max_episodes):
        init_observation = env.reset()
        init_observation = pre_processing(init_observation)
        history = np.stack((init_observation, init_observation, init_observation, init_observation), axis=2)
        history = np.reshape(history, (84, 84, 4))
        done = False
        player_score = 0
        opponent_score = 0
        while not done:
            env.render()
            action = get_action(model, history)
            real_action = action + 1
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            observation, reward, _, _ = env.step(real_action)
            observation = pre_processing(observation)
            if reward > 0:
                player_score += 1
            elif reward < 0:
                opponent_score += 1
            if player_score == 20 or opponent_score == 20:
                done = True
            next_state = pre_processing(observation)
            next_state = np.reshape([next_state], (84, 84, 1))
            next_history = np.append(next_state, history[:, :, :3], 2)
            memory.append((history, action, reward, next_history, done))
            history = next_history
            reset_counter += 1
            if reset_counter % 10 == 0:
                target_model.set_weights(model.get_weights())
            if episode_number > obs_episodes:
                train_memory_batch(memory, model, target_model)
            if done:
                if episode_number % 5 == 0:
                    model.save('my_model')
                    np.save('replay_memory.npy', memory)
                print("player's score {} at episode {}".format(player_score, episode_number))
                print("opponent's score {} at episode {}".format(opponent_score, episode_number))

train(epsilon)
