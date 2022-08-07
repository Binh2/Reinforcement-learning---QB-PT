import gym
import numpy as np
import tensorflow as tf

from tensorflow import keras
from typing import Any, List, Sequence, Tuple

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])

def render_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Render a single episode"""

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    env.render()
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    
    state, reward, done = tf_env_step(action)

    # if tf.cast(done, tf.bool):
      # break


env = gym.make("CartPole-v0")
model = keras.models.load_model("cart-pole7_ac_model")
initial_state = tf.constant(env.reset(), dtype=tf.float32)
input("Press enter to continue.")  
print("Press Ctrl + c to stop the program")
render_episode(initial_state, model, 1000000)
