import copy
import queue
import gym
import random
from utils import *
from generate_map import *


class OffSiteEnv(gym.Env):

    def __init__(self, mode="non_maze"):
        super(OffSiteEnv, self).__init__()
        self.map = None
        self.mode = mode
        self.episode = 0
        self.small_map = False if "1v1" not in mode else True
        self.map_size = {
            "non_maze": 20,
            "maze": 19,
            "non_maze1v1": 10,
            "maze1v1": 9,
        }[mode]

        # calculate something for LearningBot
        self.map_history = queue.Queue()

    def reset(self):
        """
        the difficulty(number of players) depends on trained episodes
        :return: observation
        """
        # generate map
        if self.small_map:
            self.gen_map(2)
        else:
            if self.episode < 1000:
                self.gen_map(3)
            elif 1000 <= self.episode < 5000:
                self.gen_map(random.randint(3, 5))
            else:
                self.gen_map(random.randint(3, 8))

        # observations
        self.map_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.map_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.map_history.put(copy.copy(self.view_of(PlayerColor.red)))          # LearningBot is always red
        return self.gen_observation()

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass

    def find_home(self):
        pass

    def update_map(self):
        pass

    def gen_map(self, player_num):
        if self.mode == "non_maze" or self.mode == "non_maze1v1":
            random_or_empty = random.random()
            if random_or_empty >= 0.8:
                self.map = generate_empty_map(player_num)
            else:
                self.map = generate_random_map(player_num)
        elif self.mode == "maze" or self.mode == "maze1v1":
            self.map = generate_maze_map(player_num)

    def gen_observation(self):
        return torch.cat((self.map_history.queue[0], self.map_history.queue[1], self.map_history.queue[2])).unsqueeze(0)

    def view_of(self, color):
        pass

    def visible(self, x, y):
        for i in range(4):
            tgx = x + dx[i]
            tgy = y + dy[i]
            if tgx < 1 or tgx > self.map_size or tgy < 1 or tgy > self.map_size:
                continue
            if self.map[2][tgx][tgy] == PlayerColor.red:
                return True
        return False

    def _get_colormark(self, color):
        cm = 0
        if color == PlayerColor.grey:
            cm = -40
        elif color != PlayerColor.red:
            cm = 40 + 5 * color
        return cm