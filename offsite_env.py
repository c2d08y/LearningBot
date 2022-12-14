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
        self.obs_history = queue.Queue()
        self.action_history = queue.Queue()

        # others
        self.map = None
        self.round = 0
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
        self.internal_bots = {}
        self.internal_bots_num = 0
        self.internal_bots_color = []
        self.actions_now = [[], [], [], [], [], [], [], []]

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
        self.episode += 1
        self.gen_map(self.player_num)
        self.round = 0

        # 初始化shown
        self.shown = torch.zeros([9, self.map_size, self.map_size])

        # 初始化内部bot
        self.internal_bots_num = self.player_num - 1
        for i in range(1, 9):
            if i != self.learningbot_color:
                self.internal_bots_color.append(i)
                self.internal_bots[i] = ibot.Game(i, self.get_view_of, self.bot_action_upd)

        # 处理observation
        self.obs_history.queue.clear()
        self.obs_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.obs_history.put(torch.zeros([4, self.map_size, self.map_size]))
        self.obs_history.put(copy.copy(self.get_view_of(self.learningbot_color)))

        # 先推一个空action进去 而且action_history还是只存list为妙 不然会有莫名其妙的错误
        self.action_history.queue.clear()
        self.action_history.put([-1, -1, -1, -1, -1])
        return self.gen_observation()

    def step(self, action: torch.Tensor):
        """
        执行一步
        :param action: movement => tensor([[x1, y1, x2, y2, is_half]]) 注意x,y和i,j正好相反
        :return: observation (Tensor), reward (float), done (bool), info (dict)
        """
        # 运行
        self.round += 1
        self.add_amount_crown_and_city()
        if self.round % 10 == 0:
            self.add_amount_road()

        # 执行动作
        self.execute_actions(action[0].long())

        # 生成observation
        obs = self.gen_observation()

        # 只有动作都执行完才需要检查游戏是否结束
        reward = 0
        w_state = self.win_check()
        if w_state != 0:
            reward += 300 if w_state == 2 else -300
            return obs, reward, True, {}

        # 计算上一步的奖励
        _dirx = [0, -1, 0, 1, 1, -1, 1, -1]
        _diry = [-1, 0, 1, 0, 1, -1, -1, 1]
        last_move = self.action_history.queue[-1]
        last_obs = self.obs_history.queue[-1]
        # 保存LearningBot动作 动作要先保存 要不然永远卡在这里
        if self.action_history.qsize() > 3:
            self.action_history.get()
        act = at.i_to_a(self.map_size, int(action))[0].long() - 1
        act[4] += 1
        self.action_history.put(act.long().tolist())
        # 如果动作为空
        if last_move[0] < 0 or self.round == 1:
            return obs, reward, False, {}
        # 撞山扣一点
        if last_obs[1][last_move[3] - 1][last_move[2] - 1] == BlockType.mountain:
            reward -= 0.3
        # 撞塔扣分
        if self.map[1][last_move[3] - 1][last_move[2] - 1] == BlockType.city:
            if self.map[2][last_move[3] - 1][last_move[2] - 1] != self.learningbot_color:
                reward -= 10
        # 探索新领地加分 注意 不是占领
        for i in range(8):
            t_x = last_move[3] - 1 + _dirx[i]
            t_y = last_move[2] - 1 + _diry[i]
            if t_x < 0 or t_x >= self.map_size or t_y < 0 or t_y >= self.map_size:
                continue
            if self.map[3][t_x][t_y] - last_obs[3][t_x][t_y] == 1:
                reward += explore_reward[int(self.map[1][t_x][t_y])]
                # 如果探到玩家 额外给0.01
                if self.map[3][t_x][t_y] != PlayerColor.grey:
                    reward += 0.01

        # 保存LearningBot视角地图
        if self.obs_history.qsize() > 3:
            self.obs_history.get()
        self.obs_history.put(copy.copy(self.get_view_of(self.learningbot_color)))

        return obs, reward, False, {}

    def render(self, mode="human"):
        """
        渲染 打印当前帧画面
        :param mode: human就是给人看的 要不然就不打印出来
        :return:
        """
        if mode == "human":
            print_tensor_map(self.map)
            print(f"round: {self.round}")
            time.sleep(0.65)

    def execute_actions(self, action: torch.Tensor):
        """
        执行动作
        :param action: LearningBot的动作 内置bot动作会存到类变量里边 不需要传参 注意x,y和i,j正好相反
        :return:
        """
        # 处理内置bot动作
        for i in range(self.internal_bots_num):
            cur_color = self.internal_bots_color[i]
            try:
                # 有的时候会有莫名其妙的报错 懒得调了 反正这个Bot很弱 也不差这一个回合 主要训练还是得靠和人打
                self.internal_bots[cur_color].bot_move()
            except Exception:
                continue
            if not self.actions_now[cur_color]:
                # print(f"bot {cur_color} empty move")
                continue
            cur_action = self.actions_now[cur_color]
            # 如果移动超界了 直接下一个
            skip = False
            for j in range(4):
                if not 0 <= cur_action[j] < self.map_size:
                    skip = True
                    break
            if skip:
                # print(f"skipped: {cur_action}")
                continue

            # print(f"internal bot color {cur_color}: {cur_action}")
            # 检查动作是否合法
            if self.map[2][cur_action[1]][cur_action[0]] == cur_color:
                f_amount = int(self.map[0][cur_action[1]][cur_action[0]])
                if cur_action[4] == 1:
                    mov_troop = math.ceil((f_amount + 0.5) / 2) - 1
                else:
                    mov_troop = f_amount - 1
                self.combine((cur_action[1], cur_action[0]), (cur_action[3], cur_action[2]), mov_troop)

        # 处理LearningBot动作
        act = at.i_to_a(self.map_size, int(action))[0].long().tolist()
        print(act)
        # 检查动作是否合法 act中可能会存在-1 代表空回合
        if act[0] - 1 >= 0 and self.map[2][act[1] - 1][act[0] - 1] == self.learningbot_color:
            f_amount = int(self.map[0][act[1] - 1][act[0] - 1])
            if act[4] == 1:
                mov_troop = math.ceil((f_amount + 0.5) / 2) - 1
            else:
                mov_troop = f_amount - 1
            self.combine((act[1] - 1, act[0] - 1), (act[3] - 1, act[2] - 1), mov_troop)

    def combine(self, b1: tuple, b2: tuple, cnt):
        """
        两格合并
        :param b1: from
        :param b2: target
        :param cnt: 从b1过来的兵力
        :return:
        """
        f = {
            "amount": int(self.map[0][b1[0]][b1[1]]),
            "type": int(self.map[1][b1[0]][b1[1]]),
            "color": int(self.map[2][b1[0]][b1[1]])
        }
        t = {
            "amount": int(self.map[0][b2[0]][b2[1]]),
            "type": int(self.map[1][b2[0]][b2[1]]),
            "color": int(self.map[2][b2[0]][b2[1]])
        }

        # 地形特判
        if t["type"] == BlockType.mountain:
            return

        if t["color"] == f["color"]:
            t["amount"] += cnt
            f["amount"] -= cnt
        else:
            t["amount"] -= cnt
            f["amount"] -= cnt
            if t["amount"] < 0:
                if t["type"] == BlockType.crown:
                    tcolor = t["color"]
                    for i in range(self.map_size):
                        for j in range(self.map_size):
                            if int(self.map[2][i][j]) == tcolor:
                                self.map[2][i][j] = f["color"]
                    t["type"] = BlockType.city
                t["color"] = f["color"]
                t["amount"] = -t["amount"]

        # 赋值回去
        self.map[0][b1[0]][b1[1]] = f["amount"]
        self.map[1][b1[0]][b1[1]] = f["type"]
        self.map[2][b1[0]][b1[1]] = f["color"]
        self.map[0][b2[0]][b2[1]] = t["amount"]
        self.map[1][b2[0]][b2[1]] = t["type"]
        self.map[2][b2[0]][b2[1]] = t["color"]

    def add_amount_road(self):
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[2][i][j] != PlayerColor.grey and self.map[1][i][j] == BlockType.road:
                    self.map[0][i][j] += 1

    def add_amount_crown_and_city(self):
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[2][i][j] == PlayerColor.grey:
                    continue
                if self.map[1][i][j] == BlockType.crown or self.map[1][i][j] == BlockType.city:
                    self.map[0][i][j] += 1

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
        return torch.cat((self.obs_history.queue[0], self.obs_history.queue[1], self.obs_history.queue[2])).unsqueeze(0)

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
                    if color != self.learningbot_color and int(self.shown[color][i][j]) == 0:
                        # 如果是LearningBot 那就帮它保留视野吧(●'◡'●)
                        if self.map[1][i][j] == BlockType.city or self.map[1][i][j] == BlockType.mountain:
                            map_filtered[1][i][j] = BlockType.obstacle
                    if color == self.learningbot_color:
                        # 如果是LearningBot 则还需要计算colormark
                        map_filtered[2][i][j] = self._get_colormark(PlayerColor.grey)
                else:
                    map_filtered[0][i][j] = self.map[0][i][j]
                    map_filtered[1][i][j] = self.map[1][i][j]
                    map_filtered[2][i][j] = self._get_colormark(self.map[2][i][j])
                    if color != self.learningbot_color:
                        # 只有LearningBot需要colormark
                        map_filtered[2][i][j] = self.map[2][i][j]
                self.shown[color][i][j] = vis
        return torch.cat((map_filtered, self.shown[color].unsqueeze(0)))

    def visible(self, color, x, y):
        dx9 = [0, 1, -1, 0, 1, -1, 0, 1, -1]
        dy9 = [0, 1, -1, 1, 0, 0, -1, -1, 1]
        for i in range(9):
            tgx = x + dx9[i]
            tgy = y + dy9[i]
            if tgx < 1 or tgx >= self.map_size or tgy < 1 or tgy >= self.map_size:
                continue
            if int(self.map[2][tgx][tgy]) == color:
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
            self.actions_now[color] = [None, None, None, None, None]
            self.actions_now[color][0] = action[0] - 1
            self.actions_now[color][1] = action[1] - 1
            self.actions_now[color][2] = action[2] - 1
            self.actions_now[color][3] = action[3] - 1
            self.actions_now[color][4] = action[4]
        else:
            self.actions_now[color] = False

    def win_check(self):
        """
        检查是否结束
        :return: :return: 0 -> 还在打, 1 -> bot寄了, 2 -> bot赢了
        """
        alive = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                if int(self.map[2][i][j]) == PlayerColor.grey:
                    continue
                if int(self.map[2][i][j]) not in alive:
                    alive.append(int(self.map[2][i][j]))
                if len(alive) > 1 and self.learningbot_color in alive:
                    return 0
        if alive[0] == self.learningbot_color:
            return 2
        return 1

    def quit_signal(self):
        return False if self.episode <= 10000 else True