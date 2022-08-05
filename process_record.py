import os
import pickle
import zlib
import json
import queue
import copy
from utils import *
from const import *


class Simulator(object):

    def __init__(self, game_map, players):
        self.map = copy.deepcopy(game_map)
        self.players = players
        self.winner_color = []

    def simulate(self, movements):
        """
        simulate each round and find out the winner
        :param movements: game_data
        :return:
        """
        rnd = 0
        while rnd < len(movements) - 1:
            rnd += 1
            self.map = sim_next_round(self.map, movements[rnd], rnd, self.players)

        # find out the winner's color
        for i in range(1, len(self.map)):
            for j in range(1, len(self.map)):
                if self.map[i][j]["color"] != 0 and self.map[i][j]["color"] not in self.winner_color:
                    self.winner_color.append(self.map[i][j]["color"])


class FrameGenerator(object):

    def __init__(self, game_map, players, perspective):
        self.map = copy.deepcopy(game_map)
        self.players = players
        self.perspective = perspective
        self.size = game_map[0][0]["size"]
        self.rnd = 0
        self.shown = [[self._visible(i, j) for j in range(1, self.size + 1)] for i in range(1, self.size + 1)]
        self.self_map = copy.deepcopy(game_map)
        self._fliter()

    def generate(self, movements):
        """
        generate training data
        :param movements: game_data
        :return:
        """
        history = queue.Queue()
        history.put(torch.zeros([4, self.size, self.size]))
        history.put(torch.zeros([4, self.size, self.size]))
        history.put(self._convert())
        while self.rnd < len(movements) - 1:
            # update
            self._step(movements)
            # write old observation
            self._write(history.queue[0], history.queue[1], history.queue[2],
                        self._find_player_action(movements[self.rnd]))
            # save new frame
            history.put(self._convert())
            # pop the oldest one
            if history.qsize() > 3:
                history.get()

    def _find_player_action(self, action_set):
        for player in action_set:
            # there's some weired users appeared in a game replay even they didn't join it
            try:
                if self.players[player]["color"] == self.perspective:
                    return action_set[player] if len(action_set[player]) != 0 else [0, 0, 0, 0, 0]
            except KeyError:
                pass
        return [0, 0, 0, 0, 0]

    def _step(self, movements):
        """
        go 1 round
        :param movements: movements
        :return:
        """
        self.rnd += 1
        # simulate and get full map
        self.map = sim_next_round(self.map, movements[self.rnd], self.rnd, self.players)
        # block areas out of sight
        self._fliter()

    def _convert(self):
        """
        convert self_map to tensor
        https://www.luogu.com.cn/paste/vfjtav4b
        :return:
        """
        t_t = torch.zeros([4, self.size, self.size])
        type_dict = {0: BlockType.road, 1: BlockType.crown, 2: BlockType.road, 3: BlockType.city,
                     4: BlockType.mountain, 5: BlockType.city, 10: BlockType.obstacle}
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                t_t[0][i - 1][j - 1] = self.self_map[i][j]["amount"]
                t_t[1][i - 1][j - 1] = type_dict[self.self_map[i][j]["type"]]
                t_t[2][i - 1][j - 1] = self._calc_colormark(self.self_map[i][j]["color"])
                t_t[3][i - 1][j - 1] = self.shown[i - 1][j - 1]
        return t_t

    def _calc_colormark(self, color):
        """
        generate colormark
        :param color: color
        :return:
        """
        cm = 0
        if color == PlayerColor.grey:
            cm = 0.5
        elif color > self.perspective:
            cm = color + 30
        elif color < self.perspective:
            cm = color - 30
        return cm

    def _write(self, h_0, h_1, h_2, action):
        """
        write train data in bytes
        :param h_0:
        :param h_1:
        :param h_2:
        :param action:
        :return:
        """
        state = torch.cat((h_0, h_1, h_2))
        action_t = torch.as_tensor(action)
        if self.size == 19:
            folder_s = "./Datasets/maze/state"
            folder_t = "./Datasets/maze/target"
        elif self.size == 9:
            folder_s = "./Datasets/maze1v1/state"
            folder_t = "./Datasets/maze1v1/target"
        elif self.size == 20:
            folder_s = "./Datasets/non_maze/state"
            folder_t = "./Datasets/non_maze/target"
        else:
            folder_s = "./Datasets/non_maze1v1/state"
            folder_t = "./Datasets/non_maze1v1/target"
        file_id = str(len(os.listdir(folder_s)) + 1)
        # write state
        s_f = open(os.path.join(folder_s, file_id), "wb")
        pickle.dump(state, s_f)
        s_f.close()
        # write target(action)
        t_f = open(os.path.join(folder_t, file_id), "wb")
        pickle.dump(action_t, t_f)
        t_f.close()

    def _visible(self, x, y):
        for i in range(4):
            tgx = x + dx[i]
            tgy = y + dy[i]
            if tgx < 1 or tgx > self.size or tgy < 1 or tgy > self.size:
                continue
            if self.map[tgx][tgy]["color"] == self.perspective:
                return 1
        return 0

    def _fliter(self):
        """
        call this function every round(frame)!!
        :return:
        """
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                if self._visible(i, j) == 0:
                    if self.shown[i - 1][j - 1] == 1:
                        # if it has explored before, remain its data
                        continue
                    else:
                        # if not, clear the data
                        self.self_map[i][j]["amount"] = 0
                        self.self_map[i][j]["color"] = 0
                        self.self_map[i][j]["type"] = 0
                        if 3 <= self.map[i][j]["type"] <= 5:
                            # if it's a city or a mountain, show 10(means obstacle)
                            self.self_map[i][j]["type"] = 10
                else:
                    # simply give him the data
                    self.self_map[i][j] = self.map[i][j]
                # anyway, set "shown"
                self.shown[i - 1][j - 1] = self._visible(i, j)


def is_pubg(game_map):
    """
    special judge to throw out pubg
    :param game_map:
    :return:
    """
    size = game_map[0][0]["size"]
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if game_map[i][j]["amount"] == 100:
                return True
    return False


def main():
    record_path = r"D:\MyFiles\LearningBot\records"
    records = os.listdir(record_path)
    total_cnt = 0

    for record in records:
        with open(os.path.join(record_path, record), 'rb') as r:
            game_data = json.loads(zlib.decompress(r.read(), zlib.MAX_WBITS + 16))
        gm = game_data[0]                           # game map
        actions = game_data                         # action for round i: actions[i]
        player = game_data[0][0][0]["player"]       # players, use uid to query uname and color
        if is_pubg(gm):
            continue

        try:
            sim = Simulator(gm, player)
            sim.simulate(actions)
        except Exception:
            continue
        if len(sim.winner_color) != 1:
            # sometimes other players surrender
            continue

        fg = FrameGenerator(gm, player, sim.winner_color[0])
        fg.generate(actions)
        total_cnt += 1

        if total_cnt % 50 == 0:
            print(f"{total_cnt} records generated.")


if __name__ == '__main__':
    main()
