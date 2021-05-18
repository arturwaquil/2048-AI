import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Hide TensorFlow debugging info
# (https://github.com/tensorflow/tensorflow/issues/1258#issuecomment-267777946)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import tensorflow as tf

from game import Game2048
from gui import GUI


# Define the core network architecture
def create_model(conv=False):

    convolutional = [
        # Convolutional layers to extract positional knowledge
        tf.keras.layers.Reshape((4,4,1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation="relu"),
        tf.keras.layers.MaxPool2D()
    ]

    dense = [
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),

        # Output layer
        tf.keras.layers.Dense(units=4, activation=None)
    ]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer((16,)))

    if conv:
        [ model.add(layer) for layer in convolutional ]
    
    [ model.add(layer) for layer in dense ]

    return model

# Run model over an observation (or a batch of observations) and choose an action from
# the categorical distribution defined by the log probabilities of each possible action.
# Impossible moves are disconsidered before the sampling.
def choose_action(model, observation, possible_moves, single=True):

    # Mask of possible/impossible moves. Impossible moves
    # must have value 1 to be masked in the masked array
    possible_moves = 1 - possible_moves.astype(int)

    possible_moves = np.expand_dims(possible_moves, axis=0) if single else possible_moves
    observation = np.expand_dims(observation, axis=0) if single else observation

    # If this function is used for processing a batch of observations, it will be more
    # efficient to use model.predict() instead of model.__call__()
    logits = model(observation)

    # Replace impossible actions' values with negative infinity (which, in the logits,
    # corresponds to a null probability).
    logits = np.ma.masked_array(logits, mask=possible_moves).filled(-np.inf)

    action = tf.random.categorical(logits, num_samples=1).numpy().flatten()
    return action[0] if single else action

# Structure to save the history of each episode,
# so that the model can train at the end of it
class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

# Compute return (normalized, discounted, cumulative rewards)
def discount_rewards(rewards, gamma=0.95):

    # Aux function to normalize array
    def normalize(x):
        x = x.astype(np.float32)
        x -= np.mean(x)
        x /= np.std(x)
        return x

    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)

# Execute a forward pass through the network, compute loss, and run backpropagation
def train_step(model, optimizer, observations, actions, discounted_rewards):

    # Compute loss scaling each action's probability by the reward it resulted in
    def compute_loss(logits, actions, rewards):
        neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=actions )
        loss = tf.reduce_mean( neg_logprob * rewards )
        return loss

    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Take the log2 of board with powers of 2.
def preprocess_obs(observation):
    return np.ma.log2(observation).filled(0).astype(np.float32).flatten()

# Do all the training process
def train(model, episodes=100, ckpt=None, manager=None):

    big_tic = time.time()

    game = Game2048(seed=1)
    memory = Memory()

    # Track progress
    scores = []
    highest_tiles = []
    steps_list = []

    # If ckpt and manager were passed, set flag to save training checkpoints
    save_ckpts = ckpt is not None and manager is not None

    # Aux function to print training log
    def print_data(data):
        [ print((str(item) + '\t').expandtabs(15), end='') for item in data ]
        print("")

    for episode in range(episodes):

        if episode % 10 == 0:
            print_data(["\nEpisode", "Time", "Reward", "Score", "Highest", "  L   U   R   D", "Steps\n"])

        # Reinitialize game and progress-tracking variables
        tic = time.time()
        game.new_game()
        _, observation = game.current_state()
        observation = preprocess_obs(observation)

        memory.clear()

        action_history = [0,0,0,0]
        old_score = 0
        steps = 0

        while True:

            # Select feasible action based on the model, and perform it in the game
            action = choose_action(model, observation, np.array(game.possible_moves()))

            next_observation, score, done, tiles_merged = game.step(action)

            # TODO: Rethink how the reward is obtained. Maybe getting the score at each step
            # is not the best strategy. Other possibilities are: getting the final score of
            # the game; getting the final sum of tiles; getting the difference between the
            # sum of tiles now and in previous step; or a mixture of the mentioned strategies.
            # Maybe use metrics from the preprocessed observations instead of the raw ones.

            # # Need to experiment a bit more
            # reward1 = min((score - old_score)/1024, 1)
            # reward2 = min(tiles_merged/4, 1)
            # reward = 0.7*reward1 + 0.3*reward2
            reward = score - old_score

            next_observation = preprocess_obs(next_observation)

            old_score = score

            memory.add_to_memory(observation, action, reward)
            observation = next_observation

            action_history[action] += 1
            steps += 1

            # Train model at the end of each episode
            if done:
                # Calculate total reward of the episode and store it in the history
                total_reward = sum(memory.rewards)

                scores.append(score)

                highest_tile = int(2**np.max(observation))
                highest_tiles.append(highest_tile)

                steps_list.append(steps)

                time_since_start = time.time()-big_tic
                if time_since_start < 100:
                    elapsed_time = "{:.1f}s (+{:.1f}s)".format(time_since_start, time.time()-tic)
                else:
                    elapsed_time = "{}s (+{:.1f}s)".format(int(np.round(time_since_start)), time.time()-tic)

                actions = "{:3d} {:3d} {:3d} {:3d}".format(*action_history)

                print_data([episode, elapsed_time, total_reward, score, highest_tile, actions, steps])

                # Train the model using the stored memory
                train_step(model, optimizer,
                        observations = np.vstack(memory.observations),
                        actions = np.array(memory.actions),
                        discounted_rewards = discount_rewards(memory.rewards))

                # Save training checkpoint for every tenth episode
                if save_ckpts and (episode+1) % 10 == 0:
                    save_path = manager.save()
                    # print("Saved checkpoint for episode {}: {}\n".format(episode, save_path))

                memory.clear()
                break

    big_elapsed = int(time.time() - big_tic)
    print("\nTotal training time: {}s\n".format(big_elapsed))

    return model, [scores, highest_tiles, steps_list]

def moving_average(data, window, mode='valid'):
    return np.convolve(data, np.ones(window)/window, mode)

def plot(scores, highest_tiles, steps):
    _, axes = plt.subplots(2,2)

    # Evolution of episode scores
    axes[0,0].plot(scores, label='Scores', color='orange')
    axes[0,0].plot(moving_average(np.array(scores), 10), label='Moving average', color='blue')
    axes[0,0].set_xlabel('Episodes')
    axes[0,0].legend()

    # Evolution of number of steps per episode
    axes[0,1].plot(steps, label='Steps', color='orange')
    axes[0,1].plot(moving_average(np.array(steps), 10), label='Moving average', color='blue')
    axes[0,1].set_xlabel('Episodes')
    axes[0,1].legend()

    # Evolution of highest tiles
    axes[1,0].bar(list(range(len(highest_tiles))), highest_tiles, label='Highest tiles')
    axes[1,0].set_xlabel('Episodes')
    axes[1,0].legend()

    # Histogram of highest tiles considering all episodes
    axes[1,1].hist(np.log2(highest_tiles), list(range(16)))
    axes[1,1].set_xlabel('Highest tile (log2)')

    plt.show()

# Show model running on the game's GUI
def run_model_on_gui(model, sleep=0.1):

    print("Running model on GUI... ", end="")

    gui = GUI(seed=1)
    _, observation = gui.game.current_state()

    done = False

    while not done:
        action = choose_action(model, preprocess_obs(observation), np.array(gui.game.possible_moves()))
        observation, score, done, _ = gui.game.step(action)
        gui.update_screen()
        time.sleep(sleep)

    print("Final score: {}".format(score))
    time.sleep(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restore', dest='restore', action='store_true', help='Restore latest saved checkpoint')
    parser.add_argument('-t', '--train', dest='train', action='store_true', help='Train model')
    parser.add_argument('-e', '--episodes', type=int, default='100', help='Number of training episodes')
    parser.add_argument('-c', '--convolutional', dest='conv', action='store_true', help='Set model to have convolutional layers')
    args = parser.parse_args()

    # Instantiate model and optimizer
    model = create_model(args.conv)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Create checkpoint-managing structures
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, "./training", max_to_keep=5)

    # Restore latest saved checkpoint
    if args.restore:
        ckpt.restore(manager.latest_checkpoint)

    # Execute the training process
    if args.train:
        model, info = train(model, ckpt=ckpt, manager=manager, episodes=args.episodes)
        plot(*info)

    run_model_on_gui(model, 0.005)