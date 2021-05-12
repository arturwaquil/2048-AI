import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from game import Game2048


game = Game2048(4)
game.seed(1)

n_actions = 4   # Left, up, right, down


########## Define the model ##########

def create_model():
    # TODO: Add convolutional layers...
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])

# This function can receive either a single observation or a batch thereof
def choose_action(model, observation, single=True):
    observation = np.expand_dims(observation, axis=0) if single else observation
    logits = model.predict(observation)
    action = tf.random.categorical(logits, num_samples=1).numpy().flatten()
    return action[0] if single else action


########## Define agent's memory ##########

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

memory = Memory()


########## Reward function ##########

def normalize(x):
    x = x.astype(np.float32)
    x -= np.mean(x)
    x /= np.std(x)
    return x

def discount_rewards(rewards, gamma=0.95): 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)


########## Loss function ##########

def compute_loss(logits, actions, rewards): 
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=actions )
    loss = tf.reduce_mean( neg_logprob * rewards )
    return loss


########## Training step (forward and backpropagation) ##########

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)
  
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


########## Training ##########

# Take the log2 of board with powers of 2.
def preprocess_obs(observation):
    # TODO: When convolutional layers are added, remove the flatten()
    return np.ma.log2(observation).filled(0).astype(np.float32).flatten()

def append_smoothed(smoothed, value, smoothing_factor=0.9):
    if len(smoothed)>0:
        value = smoothing_factor*smoothed[-1] + (1-smoothing_factor)*value
    smoothed.append( value )
    return smoothed

# Metaparameters
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Instantiate model
model = create_model()
model.build(input_shape=(1,16))

# History of smoothed rewards to track progress
smoothed_reward = []

for i_episode in range(100):

    # Initialize new game, get initial observation
    game.new_game()
    _, observation = game.current_state()
    observation = preprocess_obs(observation)

    memory.clear()
  
    while True:

        # Select action based on the model, and perform it in the game
        action = choose_action(model, observation)
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
            print("episode", i_episode, total_reward)
            
            # Train the model using the stored memory
            train_step(model, optimizer, 
                       observations = np.vstack(memory.observations),
                       actions = np.array(memory.actions),
                       discounted_rewards = discount_rewards(memory.rewards))
            
            memory.clear()
            break

# Plot reward evolution
plt.plot(smoothed_reward, label='Smoothed rewards')
plt.xlabel('Episodes')
plt.legend()
plt.show()