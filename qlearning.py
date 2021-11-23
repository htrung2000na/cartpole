import gym
import random as rd
import numpy as np
from collections import defaultdict
env = gym.make('CartPole-v0')
env.reset()

epsilon = 0.5
alpha = 0.01
ganma = 0.9
Q = defaultdict(lambda: np.zeros(2))

for ep in range(1000):
    total_reward = 0
    observation = env.reset()
    if observation[2] < 0:
        s = "lech trai"
    else:
        s = "lech phai"
    for i in range(1, 10000):
        env.render()
        temp = rd.randrange(100)
        if temp <= epsilon*100:
            action = rd.randrange(2)
        else:
            action = np.argmax(Q[s])
        observation, reward, done, info = env.step(action)
        if observation[2] < 0:
            s_next = "lech trai"
        else:
            s_next = "lech phai"
        Q[s][action] += alpha*(reward + ganma*np.argmax(Q[s_next] - Q[s][action]))
        s = s_next
        total_reward += reward
        if done == True:
            print(total_reward)
            break
env.close()

