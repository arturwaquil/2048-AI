import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from game import Game2048
from gui import GUI


n_actions = 4   # Left, up, right, down

# Define the core network architecture
def create_model():
    # TODO: Add convolutional layers...
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])

# Run model over an observation (or a batch of observations) and choose an action from
# the categorical distribution defined by the log probabilities of each possible action.
def choose_action(model, observation, single=True):
    observation = np.expand_dims(observation, axis=0) if single else observation
    logits = model.predict(observation)
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
    # TODO: When convolutional layers are added, remove the flatten()
    return np.ma.log2(observation).filled(0).astype(np.float32).flatten()

# Append value to list, applying a smoothing factor
def append_smoothed(smoothed, value, smoothing_factor=0.9):
    if len(smoothed)>0:
        value = smoothing_factor*smoothed[-1] + (1-smoothing_factor)*value
    smoothed.append( value )
    return smoothed

# Do all the training process
def train(model, episodes=100, ckpt=None, manager=None):

    game = Game2048(seed=1)
    memory = Memory()

    # History of smoothed rewards to track progress
    smoothed_reward = [0]

    # If ckpt and manager were passed, set flag to save training checkpoints
    save_ckpts = ckpt is not None and manager is not None

    for episode in range(episodes):

        tic = time.time()

        # Initialize new game, get initial observation
        game.new_game()
        _, observation = game.current_state()
        observation = preprocess_obs(observation)

        memory.clear()
    
        while True:

            # Select action based on the model, and perform it in the game
            action = choose_action(model, observation)
            # TODO: ignore unfeasible moves
            next_observation, reward, done = game.step(action)
            next_observation = preprocess_obs(next_observation)
            # TODO: Rethink how the reward is obtained. Maybe getting the score at each step
            # is not the best strategy. Other possibilities are: getting the final score of
            # the game; getting the final sum of tiles; getting the difference between the
            # sum of tiles now and in previous step; or a mixture of the mentioned strategies.
            # Maybe use metrics from the preprocessed observations instead of the raw ones.

            memory.add_to_memory(observation, action, reward)
            observation = next_observation
            
            # Train model at the end of each episode
            if done:
                # Calculate total reward of the episode and store it in the history
                # TODO: Understand the need of a _smoothed_ reward history. Is it only for
                # better visualizing the increments on performance?
                total_reward = sum(memory.rewards)
                smoothed_reward = append_smoothed(smoothed_reward, total_reward)
                elapsed = int(time.time() - tic)
                print("Episode {}. Time elapsed: {}s. Total reward: {}".format(episode, elapsed, total_reward))
                
                # Train the model using the stored memory
                train_step(model, optimizer, 
                        observations = np.vstack(memory.observations),
                        actions = np.array(memory.actions),
                        discounted_rewards = discount_rewards(memory.rewards))

                # Save training checkpoint for every tenth episode
                if save_ckpts:
                    ckpt.step.assign_add(1)
                    if int(ckpt.step) % 10 == 0:
                        save_path = manager.save()
                        print("Saved checkpoint for episode {}: {}".format(episode, save_path))
                
                memory.clear()
                break
    
    return model, smoothed_reward

# Show model running on the game's GUI
def run_model_on_gui(model, sleep=0.2):
    
    print("Running model on GUI... ", end="")

    gui = GUI(seed=1)
    _, observation = gui.game.current_state()

    done = False

    while not done:
        action = choose_action(model, preprocess_obs(observation))
        observation, tmp_score, done = gui.game.step(action)
        gui.update_screen()
        if tmp_score != 0:
            score = tmp_score
        time.sleep(sleep)

    print("Final score: {}".format(score))

# Create checkpoint-managing structures for training
def create_checkpoints(model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, "./training", max_to_keep=5)
    return ckpt, manager


if __name__ == "__main__":

    # Instantiate model and optimizer
    model = create_model()
    model.build(input_shape=(1,16))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Create checkpoint-managing structures
    ckpt, manager = create_checkpoints(model, optimizer)

    # Restore latest saved checkpoint
    ckpt.restore(manager.latest_checkpoint)    

    # Execute the training process
    model, smoothed_reward = train(model, ckpt=ckpt, manager=manager)

    # Plot reward evolution
    plt.plot(smoothed_reward, label='Smoothed rewards')
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()

    run_model_on_gui(model)