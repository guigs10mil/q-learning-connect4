from connectx import Agent, make_environment
import play
import train

ncols = 7
nrows = 6

num_episodes = 30000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0002

# SETUP
env = make_environment()
agent_47 = Agent(ncols, nrows, min_exploration_rate, max_exploration_rate, exploration_decay_rate, learning_rate, discount_rate)
agent_47.load_q_table("agent_47_qt_60")

# TRAINING
# for _ in range(100000):
#   rewards_all_episodes = train.train(env, agent_47, num_episodes, max_steps_per_episode, 'random')
#   train.show_training_results(num_episodes, rewards_all_episodes)
#   train.save_agent(agent_47, "agent_47_qt")

# PLAY
play.play(env, agent_47, 'negamax')
# play.play_with_player(env, agent_47)