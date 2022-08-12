import copy
import queue
import gym
import random
import bot_div.game as ibot
from generate_map import *


class OffSiteEnv(gym.Env):

    def __init__(self, mode="non_maze"):
        super(OffSiteEnv, self).__init__()

        # LearningBot相关
        self.learningbot_color = PlayerColor.red
        self.map_history = queue.Queue()

        # others
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
        self.shown = None
        self.player_num = 0
        self.internal_bots = [ibot.Game(self.get_view_of, self.bot_action_upd) for _ in range(7)]
        self.internal_bots_num = 0
        self.internal_bots_color = []
        for i in range(1, 9):
            if i != self.learningbot_color:
                self.internal_bots_color.append(i)
        self.actions_now = [0, 0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        """
        the difficulty(number of players) depends on trained episodes
        :return: observation
        """
        # 生成地图
        if self.small_map:
            self.player_num = 2
        else:
            if self.episode < 1000:
                self.player_num = 3
            elif 1000 <= self.episode < 5000:
                self.player_num = random.randint(3, 5)
            else:
                self.player_num = random.randint(3, 8)
        self.gen_map(self.player_num)

        # 初始化shown
        self.shown = torch.zeros([self.player_num, self.map_size, self.map_size])

        # 初始化内部bot
        self.internal_bots_num = self.player_num - 1
        for i in range(self.internal_bots_num):
            self.internal_bots[i].set_color(self.internal_bots_color[i])

        # 处理observation
        self.map_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.map_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.map_history.put(copy.copy(self.get_view_of(self.learningbot_color)))
        return self.gen_observation()

    def step(self, action):
        pass

    def render(self, mode="human"):
        if mode == "human":
            print_tensor_map(self.map)

    def find_home(self):
        pass

    def update_map(self, actions):
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

    def get_view_of(self, color):
        """
        get player view
        :param color:
        :return:
        """
        map_filtered = torch.zeros([3, self.map_size, self.map_size])
        for i in range(self.map_size):
            for j in range(self.map_size):
                vis = self.visible(color, i, j)
                if not vis:
                    # 如果这个玩家现在看不到这一格
                    if color != self.learningbot_color or int(self.shown[color][i][j]) == 0:
                        # 如果是LearningBot 那就帮它保留视野吧(●'◡'●)
                        if self.map[1][i][j] == BlockType.city:
                            map_filtered[1][i][j] = BlockType.obstacle
                else:
                    map_filtered[0][i][j] = self.map[0][i][j]
                    map_filtered[1][i][j] = self.map[1][i][j]
                    map_filtered[2][i][j] = self._get_colormark(self.map[2][i][j])
                self.shown[color][i][j] = vis
        return torch.cat((map_filtered, self.shown[color].unsqueeze()))

    def visible(self, color, x, y):
        for i in range(4):
            tgx = x + dx[i]
            tgy = y + dy[i]
            if tgx < 1 or tgx > self.map_size or tgy < 1 or tgy > self.map_size:
                continue
            if self.map[2][tgx][tgy] == color:
                return True
        return False

    def _get_colormark(self, color):
        cm = 0
        if color == PlayerColor.grey:
            cm = -40
        elif color != self.learningbot_color:
            cm = 40 + 5 * color
        return cm

    def bot_action_upd(self, color, action):
        """
        收集内部bot的动作
        :param color:
        :param action: [x1, y1, x2, y2, is_half]
        :return:
        """
        if action:
            self.actions_now[color] = torch.LongTensor(action)
        else:
            self.actions_now[color] = torch.LongTensor([0, 0, 0, 0, 0])