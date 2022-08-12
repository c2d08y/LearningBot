import copy
import queue
import re
import gym
import torch.distributed.optim
from selenium.webdriver import ActionChains
from utils import *
from const import *
import settings


class OnSiteEnv(gym.Env):

    def __init__(self):
        super(OnSiteEnv, self).__init__()

        # 常量
        self.base_url = "https://kana.byha.top:444"
        self.room = settings.default_room
        self.block_finder = re.compile(r'<td.*?id="td-\d+".*?class="(.*?)".*?>([\d\s]*)</td>')

        # 初始化浏览器
        self.driver = webdriver.Chrome(options=init_driver_options())
        login(self.driver)

        # 游戏局内参数
        self.game_table = None
        self.map = None
        self.map_size = None
        self.self_color = None
        self.selected = (0, 0)
        self.map_history = queue.Queue()        # 只读 整型
        self.action_history = queue.Queue()     # 只读
        self.view = False
        self.observation = None
        self.crown_ele = None
        self.shown_before = None

        # 全局临时变量
        self._map_data = None
        self._blocks = None

    def reset(self):
        """
        重设 顺便等待游戏开始
        :return: observation (Tensor)
        """
        if self.driver.current_url != self.base_url + "/checkmate/room/" + self.room:
            # 如果不在房间内
            self.enter_room(self.room)
        if self.view:
            # 如果是旁观
            self.driver.find_element(By.ID, "view").click()
        # 准备
        ActionChains(self.driver).click(self.driver.find_element_by_id("ready")).perform()
        # 等
        WebDriverWait(self.driver, 86400).until_not(EC.text_to_be_present_in_element((By.ID, "game-status"), "准备中"))
        # 获取table
        self.game_table = self.driver.find_element(By.ID, "m")
        # 初始化地图
        c1, c2 = self.init_map()
        # 保存action
        self.action_history.put(torch.as_tensor([0, 0, 0, 0, 0], dtype=torch.long))

        if c1 != 1 or c2 >= 100:
            # 如果是流浪或者抓虾 那没事了
            self.driver.find_element(By.ID, "view").click()
            self.view = True
        return self.observation

    def step(self, action: torch.Tensor):
        """
        执行一步
        :param action: movement => tensor([[x1, y1, x2, y2, is_half]])
        :return: observation (Tensor), reward (float), done (bool), info (dict)
        """
        reward = 0

        # 执行到这里其实还是上一步 等下一回合
        wait_until(next_round(self.crown_ele.text), self.driver)

        # 先看看游戏是否结束
        state_now = self.win_check()
        if state_now != 0:
            reward = 100 if state_now == 2 else -100
            return self.observation, reward, True, {}

        self.update_map()
        try:
            # 这里有可能也会结束 因为move比较耗时
            self.move(action)
        except Exception:
            state_now = self.win_check()
            if state_now != 0:
                reward = 100 if state_now == 2 else -100
                return self.observation, reward, True, {}

        # 计算这一步的奖励
        _dirx = [0, -1, 0, 1, 1, -1, 1, -1]
        _diry = [-1, 0, 1, 0, 1, -1, -1, 1]
        last_move = self.action_history.queue[-1]
        last_map = self.map_history.queue[-1]
        # 无效移动扣大分
        if last_map[2][last_move[0] - 1][last_move[1] - 1] != self._get_colormark(self.self_color):
            reward -= 100
        # 撞塔扣分
        if self.map[1][last_move[2] - 1][last_move[3] - 1] == BlockType.city:
            if self.map[2][last_move[2] - 1][last_move[3] - 1] != self._get_colormark(self.self_color):
                reward -= 10
        # 探索新领地加分 注意 不是占领
        for i in range(8):
            t_x = last_move[2] - 1 + _dirx[i]
            t_y = last_move[3] - 1 + _diry[i]
            if t_x < 0 or t_x >= self.map_size or t_y < 0 or t_y >= self.map_size:
                continue
            if self.map[3][t_x][t_y] - last_map[3][t_x][t_y] == 1:
                reward += explore_reward[self.map[1][t_x][t_y]]
                # 如果探到玩家 额外给0.5
                if self.map[3][t_x][t_y] != self._get_colormark(PlayerColor.grey):
                    reward += 0.5

        # 保存action
        if self.action_history.qsize() == 3:
            self.action_history.get()
        self.action_history.put(copy.copy(action[0].long()))

        # 再检查一遍 有没有结束
        state_now = self.win_check()
        if state_now != 0:
            reward = 100 if state_now == 2 else -100
            return self.observation, reward, True, {}

        return self.observation, reward, False, {}

    def render(self, mode="human"):
        """
        在网站上玩为什么需要渲染 =_=
        :param mode:
        :return:
        """
        pass

    def init_map(self):
        time.sleep(0.2)
        # 获取地图大小
        self._map_data = self.game_table.get_attribute("innerHTML")
        self._blocks = self.block_finder.findall(self._map_data)
        self.map_size = int(math.sqrt(len(self._blocks)))
        # 找自己家
        crown_s = self.driver.find_elements(By.CSS_SELECTOR, ".own.crown")
        self.crown_ele = crown_s[0]
        self.self_color = self._get_color(self.crown_ele.get_attribute("class"))
        cnt1 = len(crown_s)
        cnt2 = int(self.crown_ele.text)
        # 初始化地图和shown标记
        self.map = torch.zeros([4, self.map_size, self.map_size])
        self.shown_before = torch.zeros([self.map_size, self.map_size])
        for i in range(3):
            self.map_history.put(copy.copy(self.map))
        self.update_map(True)
        return cnt1, cnt2

    def update_map(self, _init_flag=False):
        """
        更新地图 顺便更新observation
        :param _init_flag: 如果init_map叫我 那就是True
        :return:
        """
        # 弹出旧地图 压入新地图
        # 注意 self.map_history中的数据是只读
        if self.map_history.qsize() == 3:
            self.map_history.get()
        self.map_history.put(copy.copy(self.map))

        if not _init_flag:
            # init_map会顺便帮我整这些东西的
            self._map_data = self.game_table.get_attribute("innerHTML")
            # self._blocks[index][0] means class name
            # self._blocks[index][1] means value
            self._blocks = self.block_finder.findall(self._map_data)
        # 用于遍历self._blocks
        index = 0
        for i in range(self.map_size):
            for j in range(self.map_size):
                try:
                    # 获取这一格上的兵力
                    b_value = int(self._blocks[index][1])
                except ValueError:
                    # 如果是空的 会爆ValueError
                    b_value = 0
                # get class name
                b_attr = self._blocks[index][0]
                # 看看是否在视野内
                shown = False if "unshown" in b_attr else True
                if int(self.shown_before[i][j]) == 1 and shown == 0:
                    # 如果以前看到过 保留视野 但shown标记跟随地图
                    self.map[3][i][j] = shown
                    continue
                if shown:
                    # 只要看到过 就标成1
                    self.shown_before[i][j] = 1
                # 获取兵力和类型
                if "unshown" == b_attr:
                    self.map[0][i][j] = b_value
                    self.map[1][i][j] = BlockType.road
                    b_attr += " grey"
                elif "null" in b_attr:
                    self.map[0][i][j] = b_value
                    self.map[1][i][j] = BlockType.road
                elif "obstacle" in b_attr:
                    self.map[0][i][j] = -1
                    self.map[1][i][j] = BlockType.obstacle
                    b_attr += " grey"
                elif "mountain" in b_attr:
                    self.map[0][i][j] = -1
                    self.map[1][i][j] = BlockType.mountain
                    b_attr += " grey"
                elif "crown" in b_attr:
                    self.map[0][i][j] = b_value
                    self.map[1][i][j] = BlockType.crown
                elif "city" in b_attr:
                    self.map[0][i][j] = b_value
                    self.map[1][i][j] = BlockType.city
                elif "empty-city" in b_attr:
                    self.map[0][i][j] = b_value
                    self.map[1][i][j] = BlockType.city
                    b_attr += " grey"
                # get colormark
                color = self._get_color(b_attr)
                self.map[2][i][j] = self._get_colormark(color)
                # set shown
                self.map[3][i][j] = shown

                index += 1

        # 三帧并在一起作为observation
        self.observation = torch.cat((self.map_history.queue[0], self.map_history.queue[1], self.map_history.queue[2]))
        self.observation = self.observation.unsqueeze(0)

    def move(self, mov):
        """
        just as the name
        :param mov: tensor([[x1, y1, x2, y2, is_half]])
        :return:
        """
        move_info = mov[0].long()
        if self.selected[0] != move_info[0] - 1 or self.selected[1] != move_info[1] - 1:
            # 如果没选中 先点一下
            self.driver.find_element_by_id(f"td-{int((move_info[0] - 1) * self.map_size + move_info[1])}").click()

        # 获取移动方向 决定按哪个键
        keys = ['W', 'A', 'S', 'D']
        difx = move_info[2] - move_info[0]
        dify = move_info[3] - move_info[1]
        for i in range(4):
            if difx == dx[i] and dify == dy[i]:
                ActionChains(self.driver).send_keys(keys[i]).perform()

        if self.map[1][move_info[2] - 1][move_info[3] - 1] != BlockType.mountain and \
                self.map[1][move_info[2] - 1][move_info[3] - 1] != BlockType.obstacle:
            self.selected = (move_info[2] - 1, move_info[3] - 1)

    def win_check(self) -> int:
        """
        虽然也有可能是输了
        :return: 0 -> 还在打, 1 -> bot寄了, 2 -> bot赢了
        """
        try:
            t = self.driver.find_element(By.ID, "swal2-content")
            if t.text.strip() == settings.bot_name + "赢了":
                return 2
        except NoSuchElementException:
            return 0
        return 1

    def enter_room(self, room_id: str):
        """
        进房间
        :param room_id: room name
        :return:
        """
        self.driver.get(self.base_url + "/checkmate/room/" + room_id)
        self.room = room_id
        self.game_table = self.driver.find_element_by_id("m")

    def _get_color(self, class_name: list) -> int:
        if "grey" in class_name:
            return PlayerColor.grey
        if "blue" in class_name:
            return PlayerColor.blue
        if "red" in class_name:
            return PlayerColor.red
        if "green" in class_name:
            return PlayerColor.green
        if "orange" in class_name:
            return PlayerColor.orange
        if "pink" in class_name:
            return PlayerColor.pink
        if "purple" in class_name:
            return PlayerColor.purple
        if "chocolate" in class_name:
            return PlayerColor.chocolate
        if "maroon" in class_name:
            return PlayerColor.maroon
        return PlayerColor.grey

    def _get_colormark(self, color):
        cm = 0
        if color == PlayerColor.grey:
            cm = -40
        elif color != self.self_color:
            cm = 40 + 5 * color
        return cm

    def quit_signal(self):
        return False