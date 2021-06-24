import numpy as np
import os
import time
import logging as log
from connectx import Agent

log.basicConfig(filename='train.log', encoding='utf-8', format='[%(levelname)s] %(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=log.DEBUG)

def train(env, agent: Agent, num_episodes, max_steps_per_episode, opponent='random'):
  # Define training agent
  trainer = env.train([None, opponent])

  rewards_all_episodes = []

  start_time = time.time()
  last_measure = start_time

  # Q-learning algorithm
  for episode in range(num_episodes):
    state = trainer.reset()
    board = state['board']

    done = False

    rewards_current_episode = 0
    agent.reset_valid_actions()

    for step in range(max_steps_per_episode):

      # Generate action
      action = agent.action(board)

      if type(action) != int:
        log.debug(type(action))
        log.critical("Sorry, no numbers below zero")

      # Take a step
      new_obs, reward, done, info = trainer.step(action)

      # Update Q-table
      new_board = new_obs['board']
      agent.update_q_table(new_board, reward, action, board)

      # Set new state
      board = new_board

      # Add new reward
      rewards_current_episode += reward if reward else 0

      if done:
        break  

    # Exploration rate decay
    agent.update_exploration_rate(episode)

    if not (episode % (num_episodes/10)):
      log.debug(f"episode: {episode} - time spent: {time.strftime('%H:%M:%S', time.gmtime(time.time() - last_measure))}")
      last_measure = time.time()

    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

  log.info(f"Total time spent: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

  return rewards_all_episodes


def show_training_results(num_episodes, rewards_all_episodes):
  i = (num_episodes/10)

  # Calculate and log.debug the average reward
  rewards_per_n_episodes = np.split(np.array(rewards_all_episodes),num_episodes/i)
  count = i

  log.debug(f"Average reward per {int(i)} episodes")
  for r in rewards_per_n_episodes:
    r = list(filter(None, r))
    log.debug(f"{count}: {str(sum(r)/i)}")
    count += i

def save_agent(agent: Agent, filename="q_table"):
  files = os.listdir('.')
  for i in range(100000):
    new_file = f"{filename}_{i}"
    if new_file not in files:
      agent.save_q_table(new_file)
      log.info(f'Saved file: {new_file}')
      break