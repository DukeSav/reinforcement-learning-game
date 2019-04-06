import gym
import numpy as np
import random
import cv2
import keras


from replay_buffer import ReplayBuffer

from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

# List of hyper-parameters and constants used in the deeepQ class ( Brain of the model)
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 6
TAU = 0.01

# List of hyper-parameters and constants used in the Space Invaders class
BUFFER_SIZE_T = 100000
MINIBATCH_SIZE_T = 32
TOT_FRAME_T = 1000000
EPSILON_DECAY_T = 300000
MIN_OBSERVATION_T = 5000
FINAL_EPSILON_T = 0.1
INITIAL_EPSILON_T = 1.0

# Number of frames to throw into network
NUM_FRAMES = 3

# The brain of our game


class DeepQ(object):
    """Constructs the desired deep q learning network"""

    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        #define model as sequential
        self.model = Sequential()
        #adding convolutional2D layers
        #Filter Number:32, Kernel Size:8x8 Input Image:84x84x3 
        #Output is 84x84x32
        self.model.add(Convolution2D(32, 8, 8, subsample=(
            4, 4), input_shape=(84, 84, NUM_FRAMES)))
        #Activation function is relu ( linear for x>0)
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        #Flattens the model, instead of 3 parameters (84,84,64) it had their pollaplasiasmos
        self.model.add(Flatten())
        #regular layer NN
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        #compile with Adam, loss rate 0.00001 and loss = mse (mean squared error)
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper
        #creating a second neurwniko diktuo for storing the weights
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4),
                                            input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        #storing the weights
        self.target_model.set_weights(self.model.get_weights())

        print("Successfully constructed networks.")

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        #Generates output predictions for the input samples.
        q_actions = self.model.predict(
            data.reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        #if random_number < epsilon pare allo action
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        #arxikopoioume me midenika tous pinakes
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            #predict sto model
            targets[i] = self.model.predict(
                s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            #predict sto target_model
            fut_action = self.target_model.predict(
                s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size=1)
            #wtf is this plz help
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + \
                (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

# The class of our game


class SpaceInvader(object):

    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE_T)

        # Construct appropriate network
        self.deep_q = DeepQ()

        # A buffer that keeps the last 3 images
        self.process_buffer = []
        # Initialize buffer with the first frame
        s1, r1, _, _ = self.env.step(0)
        s2, r2, _, _ = self.env.step(0)
        s3, r3, _, _ = self.env.step(0)
        self.process_buffer = [s1, s2, s3]

    def load_network(self, path):
        self.deep_q.load_network(path)

    def convert_process_buffer(self):
        """ Basic image processing which converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90))
                        for x in self.process_buffer]
        black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]
        return np.concatenate(black_buffer, axis=2)

    def train(self, num_frames):
        observation_num = 0
        curr_state = self.convert_process_buffer()
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" % observation_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON_T:
                epsilon -= (INITIAL_EPSILON_T - FINAL_EPSILON_T) / \
                    EPSILON_DECAY_T

            initial_state = self.convert_process_buffer()
            self.process_buffer = []

            predict_movement, predict_q_value = self.deep_q.predict_movement(
                curr_state, epsilon)

            reward, done = 0, False
            for i in range(NUM_FRAMES):
                temp_observation, temp_reward, temp_done, _ = self.env.step(
                    predict_movement)
                reward += temp_reward
                self.process_buffer.append(temp_observation)
                done = done | temp_done

            if observation_num % 10 == 0:
                print("We predicted a q value of ", predict_q_value)

            if done:
                print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                self.env.reset()
                alive_frame = 0
                total_reward = 0

            new_state = self.convert_process_buffer()
            self.replay_buffer.add(
                initial_state, predict_movement, reward, done, new_state)
            total_reward += reward

            if self.replay_buffer.size() > MIN_OBSERVATION_T:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(
                    MINIBATCH_SIZE_T)
                self.deep_q.train(s_batch, a_batch, r_batch,
                                  d_batch, s2_batch, observation_num)
                self.deep_q.target_train()

            # Save the network every 100000 iterations
            # if observation_num % 10000 == 9999:
            if observation_num % 10000 == 0:
                print("Saving Network")
                self.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1

    def simulate(self, path="", save=False):
        """Simulates game"""
        done = False
        tot_award = 0
        if save:
            self.env = gym.wrappers.Monitor(
                self.env, directory=path, force=True, write_upon_reset=True)
        self.env.reset()
        self.env.render()
        while not done:
            state = self.convert_process_buffer()
            predict_movement = self.deep_q.predict_movement(state, 0)[0]
            self.env.render()
            observation, reward, done, _ = self.env.step(predict_movement)
            tot_award += reward
            self.process_buffer.append(observation)
            self.process_buffer = self.process_buffer[1:]
        if save:
            self.env.env.close()

    def calculate_mean(self, num_samples=100):
        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_samples):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.convert_process_buffer()
                predict_movement = self.deep_q.predict_movement(state, 0.0)[0]
                observation, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer.append(observation)
                self.process_buffer = self.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)
        return np.mean(reward_list), np.std(reward_list)


if __name__ == '__main__':
    game_instance = SpaceInvader()
    game_instance.train(10000)
    stat = game_instance.calculate_mean()
    print("Game Statistics")
    print(stat)
    game_instance.simulate("saved.h5", save=True)
    game_instance.simulate()
    # Load an already trained model and test it
    # game_instance.load_network("saved.h5")
    # while 1:
    # game_instance.simulate()
