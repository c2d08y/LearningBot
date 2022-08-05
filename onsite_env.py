import copy
import queue
import re
import gym
from selenium.webdriver import ActionChains
from utils import *
from const import *
import settings


class OnSiteEnv(gym.Env):

    def __init__(self):
        super(OnSiteEnv, self).__init__()

        # consts
        self.base_url = "https://kana.byha.top:444"
        self.room = settings.default_room
        self.block_finder = re.compile(r'<td.*?id="td-\d+".*?class="(.*?)".*?>([\d\s]*)</td>')

        # init driver
        self.driver = webdriver.Chrome(options=init_driver_options())
        login(self.driver)

        # game args
        self.game_table = None
        self.map = None
        self.map_size = None
        self.self_color = None
        self.selected = (0, 0)
        self.map_history = queue.Queue()        # readonly
        self.action_history = queue.Queue()     # readonly
        self.view = False
        self.observation = None

        # global temp variable
        self._map_data = None
        self._blocks = None

    def reset(self):
        """
        reset the environment (wait until game starts)
        :return: observation (Tensor)
        """
        if self.view:
            # if bot is in mode "view"
            self.driver.find_element(By.ID, "view").click()
        if self.driver.current_url != self.base_url + "/checkmate/room/" + self.room:
            # if bot is out of room
            self.enter_room(self.room)
        # wait game to start
        WebDriverWait(self.driver, 86400).until(EC.text_to_be_present_in_element((By.ID, "game-status"), "准备中"))
        # read table
        self.game_table = self.driver.find_element(By.ID, "m")
        # init map
        c1, c2 = self.init_map()

        if c1 != 1 or c2 >= 100:
            # surrender if game mode is "pubg" or "shrimp grabbing"
            self.driver.find_element(By.ID, "view").click()
            self.view = True
        return self.map

    def step(self, action: torch.Tensor):
        """
        do a step
        :param action: movement => [x1, y1, x2, y2, is_half]
        :return: observation (Tensor), reward (float), done (bool), info (dict)
        """
        reward = 0

        # check game state
        state_now = self.win_check()
        if state_now != 0:
            reward = 100 if state_now == 2 else -100
            return self.observation, reward, True, {}

        self.update_map()
        self.move(action)

        # calc reward
        last_move = self.action_history.queue[-1]
        if self.map[1][last_move[2]][last_move[3]] == BlockType.city:
            if self.map[2][last_move[2]][last_move[3]] != self.self_color:
                # if bot run into a tower
                reward = -10

        # save action
        if self.action_history.qsize() == 3:
            self.action_history.get()
        self.action_history.put(copy.copy(action))

        return self.observation, reward, False, {}

    def render(self, mode="human"):
        """
        on site mode do not need function "render"
        :param mode:
        :return:
        """
        pass

    def init_map(self):
        # get map size
        self._map_data = self.game_table.get_attribute("innerHTML")
        self._blocks = self.block_finder.findall(self._map_data)
        self.map_size = int(math.sqrt(len(self._blocks)))
        # search for crown
        index = 0
        cnt1 = 0
        cnt2 = 0
        for i in range(self.map_size):
            for j in range(self.map_size):
                if "crown" in self._blocks[index][0]:
                    self.self_color = self._get_color(self._blocks[index][0])
                    self.selected = (i, j)
                    # there's more than 1 crown in mode "shrimp grabbing"
                    cnt1 += 1
                    cnt2 = self._blocks[index][1]
                index += 1
        # init map
        self.map = torch.zeros([4, self.map_size, self.map_size])
        self.update_map(True)
        return cnt1, cnt2

    def update_map(self, _init_flag=False):
        """
        update map, also update observation
        :param _init_flag: True if self.init_map() call this method
        :return:
        """
        # pop oldest map and push old map
        # BE CAREFUL! Don't change data in self.map_history!
        if self.map_history.qsize() == 3:
            self.map_history.get()
        self.map_history.put(copy.copy(self.map))

        if not _init_flag:
            # init func will get these below for us
            self._map_data = self.game_table.get_attribute("innerHTML")
            # self._blocks[index][0] means class name
            # self._blocks[index][1] means value
            self._blocks = self.block_finder.findall(self._map_data)
        # to traversal self._blocks
        index = 0
        for i in range(self.map_size):
            for j in range(self.map_size):
                try:
                    # get value on that block
                    b_value = int(self._blocks[index][1])
                except ValueError:
                    # if value is 0, it causes error
                    b_value = 0
                # get class name
                b_attr = self._blocks[index][0]
                # check if it's shown
                shown = False if "unshown" in b_attr else True
                if self.map[3][i][j] == 1 and shown == 0:
                    # if it has explored before, remain its data, but set shown anyway
                    self.map[3][i][j] = shown
                    continue
                # get value and type
                if b_attr == "unshown":
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

        # combine 3 frames as observation
        self.observation = torch.cat((self.map_history.queue[0], self.map_history.queue[1], self.map_history.queue[2]))

    def move(self, move_info):
        """
        just as the name
        :param move_info: [x1, y1, x2, y2, is_half]
        :return:
        """
        if self.selected[0] != move_info[0] or self.selected[1] != move_info[1]:
            # if the block is not selected, then select it
            self.driver.find_element_by_id(f"td-{(move_info[0] - 1) * self.map_size + move_info[1]}").click()

        # get the direction and press
        keys = ['w', 'a', 's', 'd']
        difx = move_info[2] - move_info[0]
        dify = move_info[3] - move_info[1]
        for i in range(4):
            if difx == dx[i] and dify == dy[i]:
                ActionChains(self.driver).send_keys(keys[i]).perform()

        if self.map[1][move_info[2]][move_info[3]] != BlockType.mountain and \
                self.map[1][move_info[2]][move_info[3]] != BlockType.obstacle:
            self.selected = (move_info[2], move_info[3])

    def win_check(self) -> int:
        """
        also it can be lost :D
        :return: 0 -> game still going on, 1 -> bot lose, 2 -> bot win
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
        just as the name
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

    def _get_colormark(self, color):
        cm = 0
        if color == PlayerColor.grey:
            cm = 0.5
        elif color > self.self_color:
            cm = color + 30
        elif color < self.self_color:
            cm = color - 30
        return cm
