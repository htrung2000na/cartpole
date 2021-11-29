import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque 

optimizer = keras.optimizers.Adam(learning_rate=0.01)
epsilon = 0.9
batch_size = 10
ganma = 0.99
C = 10
reset = 0

env = gym.make("CartPole-v1")

D = deque(maxlen=10000)

inputs = layers.Input(shape=((5),))
hidden_layer_1 = layers.Dense(64, activation='relu')(inputs)
hidden_layer_2 = layers.Dense(32, activation='relu')(hidden_layer_1)
outputs = layers.Dense(1)(hidden_layer_2)

model = keras.Model(inputs= inputs, outputs= outputs)
target = keras.models.clone_model(model)

final_reward = 0
with tf.GradientTape(persistent=True) as tape:
    while final_reward < 100:
        state = env.reset()
        total_reward = 0
        for i in range(1000):
            env.render()
            choose_action = random.randrange(1, 101)
            if choose_action < epsilon*100:
                action = random.randrange(0, 2)
            else:
                input0 = state.copy().astype(float)
                input1 = state.copy().astype(float)
                input0 = np.append(input0, np.array(0))
                input1 = np.append(input1, np.array(1))
                input0 = np.expand_dims(input0, 0)
                input1 = np.expand_dims(input1, 0)
                if model(input0) > model(input1):
                    action = 0
                else:
                    action = 1
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            D.append((state, action, reward, next_state, done))
            if len(D) >= batch_size:
                mini_batch = random.choices(D, k = batch_size)
                for eval in mini_batch:
                    input0 = eval[0].copy().astype(float)
                    input1 = eval[0].copy().astype(float)
                    input0 = np.append(input0, np.array(0))
                    input1 = np.append(input1, np.array(1))
                    input0 = np.expand_dims(input0, 0)
                    input1 = np.expand_dims(input1, 0)
                    
                    if eval[4] == True:
                        y = eval[2]
                    else:
                        if target(input0) > target(input1):
                            y = eval[2] + ganma*target(input0)
                        else:
                            y = eval[2] + ganma*target(input1)
                    
                    inputEval = eval[0].copy().astype(float)
                    inputEval = np.append(inputEval, np.array(eval[1]))
                    inputEval = np.expand_dims(inputEval, 0)
                    loss = (y - model(inputEval))**2
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                reset += 1
                if reset >= C:
                    reset = 0
                    target = keras.models.clone_model(model)
            if done == True:
                print(total_reward)
                if total_reward > final_reward:
                    final_reward = total_reward
                break

model.save('my_model')

                

