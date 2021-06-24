import re
import matplotlib.pyplot as plt

def get_results(file):
  f = open(file)
  iter = re.finditer(r"(?<=30000.0:\s)0\.\d*", f.read())
  f.close()

  res = []
  for i in iter:
    res.append(float(i.group()))

  return res

def display_results(results):
  plt.plot(results)
  plt.ylabel("Average reward of the last 3000 episodes of each version")
  plt.xlabel("Q Agent Version")
  plt.title("Q Agent's Training Results Against 'Random'")
  plt.grid(True)
  plt.show()


a = get_results("./train.log")
# print(a)
display_results(a)