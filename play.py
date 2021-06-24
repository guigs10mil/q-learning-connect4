from utils import show_board
from connectx import Agent

def play(env, agent: Agent, opponent='random'):
  total_rewards = 0
  for i in range(100):
    agent.reset_valid_actions()
    result = env.run([agent.play, opponent])
    total_rewards += result[-1][0].reward

    if i%10 == 9:
      show_board(result[-1][0]['observation']['board'], agent.nrows, agent.ncols)
      print(result[-1][0].reward)
      print()

  print("total_rewards:", total_rewards)

def play_with_player(env, agent: Agent):
  env.render(mode="human")
  env.play([agent.play, None])