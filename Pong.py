import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque 
from tensorflow.keras.optimizers import Adam

max_episode = 200
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = epsilon_min/epsilon
epsilon_decay = epsilon_decay**(1. / float(max_episode))
batch_size = 10
ganma = 0.95
reset_counter = 0
save_counter = 0
load_model = False

env = gym.make("Pong-v0")

input_shape = (210, 160, 3)

#tạo nerual network. 4 đầu vào thể hiện trạng thái của môi trường. 2 đầu ra thể hiện giá trị Q của mỗi hành
#động
inputs = layers.Input(shape=input_shape)
hidden_layer_1 = layers.Conv2D(32, 8, strides=(4, 4), 
                              activation='relu')(inputs)
hidden_layer_2 = layers.Conv2D(64, 4, strides=(2, 2),
                              activation='relu')(hidden_layer_1)
hidden_layer_3 = layers.Conv2D(64, 3, strides=(1, 1),
                              activation='relu')(hidden_layer_2)
hidden_layer_4 = layers.Flatten()(hidden_layer_3)
hidden_layer_5 = layers.Dense(512, activation='relu')(hidden_layer_4)
outputs = layers.Dense(3, activation='linear')(hidden_layer_5)

if load_model == True:
    model = keras.models.load_model('my_model')
    target = keras.models.load_model('my_model')
    #D là replay memory.
    D = np.load('replay_memory.npy', allow_pickle=True)
    D = deque(D, maxlen=10000)
else:
    model = keras.Model(inputs= inputs, outputs= outputs)
    target = keras.Model(inputs= inputs, outputs= outputs)
    model.compile(loss='mse', optimizer=Adam())
    D = deque(maxlen=10000)

num_of_eps = 0
while True:
    #reset môi trường. Hàm reset trả về trạng thái ban đầu của môi trường
    state = env.reset()
    player_score = 0
    opponent_score = 0
    for i in range(10000):
        env.render()
        #chọn hành động theo chính sách epsilon-greedy.
        #lấy ngẫu nhiên giá trị từ 0 đến 1, nếu nhỏ hơn epsilon thì chọn hành động ngẫu nhiên
        if np.random.rand() < epsilon:
            action = random.randrange(3)
        else:
            #nếu không thì chọn hành động với giá trị Q lớn nhất
            action = np.argmax(model.predict(np.expand_dims(state, 0))[0])
        if action == 1:
            action = 3
        #thực hiện hành động vừa chọn. hàm step trả về trạng thái tiếp theo, phần thường
        #nhận được khi chọn hành động, done cho biết con lắc đã mất cân bằng chưa
        next_state, reward, done, _ = env.step(action)
        if reward > 0:
            player_score += 1
        elif reward < 0:
            opponent_score += 1
        if player_score == 20 or opponent_score == 20:
            done = True
        #Đưa dữ liệu vừa quan sát được vào D
        if action == 3:
            action = 1
        D.append((state, action, reward, next_state, done))
        state = next_state
        #lấy dữ liệu từ D dưới dạng mini batch để huấn luyện nerual network
        if len(D) >= batch_size:
            state_batch = []
            q_value_batch = []
            #lấy ngẫu nhiên 1 mini batch từ D
            mini_batch = random.sample(D, k = batch_size)
            for eval in mini_batch:
                y = model.predict(np.expand_dims(eval[0], 0))
                #công thức
                if eval[4] == True:
                    y[0][eval[1]] = eval[2]
                else:
                    y[0][eval[1]] = eval[2] + ganma*np.amax(target.predict(np.expand_dims(eval[3], 0))[0])

                state_batch.append(eval[0])
                q_value_batch.append(y[0])
            #huấn luyện DQN
            model.fit(np.array(state_batch), np.array(q_value_batch), batch_size=batch_size, epochs=1, verbose=0)
            reset_counter += 1
            #cập nhập target nerual network sau mỗi 10 bước thời gian
            if reset_counter % 10 == 0:
                target.set_weights(model.get_weights())
        if done == True:
            num_of_eps += 1
            #lưu model
            if num_of_eps % 10 == 0:
                model.save('my_model')
                np.save('replay_memory.npy', D)
            #giảm giá trị epsilon để khi huấn luyện càng lâu, xác suất
            #khám phá càng ít và xác suất khai thác càng nhiều
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            print("player's score: {:.2f} at episode {}".format(player_score, num_of_eps))
            print("opponent's score: {:.2f} at episode {}".format(opponent_score, num_of_eps))
            break

model.save('my_model')

                
