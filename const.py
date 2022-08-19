import torch


class BlockType(object):
    road = 0                # null, unshown null
    obstacle = 1            # obstacle
    mountain = 2            # mountain
    crown = 300             # crown
    city = 4                # empty-city, city


class PlayerColor(object):
    grey = 0
    blue = 1
    red = 2
    green = 3
    orange = 4
    pink = 5
    purple = 6
    chocolate = 7
    maroon = 8


explore_reward = {BlockType.road: 0.01, BlockType.mountain: 0.01, BlockType.crown: 10, BlockType.city: 0.01}


directions = {'W': (0, -1), 'S': (0, 1), 'A': (-1, 0), 'D': (1, 0)}


class FrontColor(object):
    black = 30
    red = 31
    green = 32
    yellow = 33
    blue = 34
    purple = 35
    ultramarine = 36
    white = 37


class BackgroundColor(object):
    black = 40
    red = 41
    green = 42
    yellow = 43
    blue = 44
    purple = 45
    ultramarine = 46
    white = 47


class Style(object):
    default = 0
    high_light = 1
    italic = 3
    under_line = 4
    twinkle = 5
    anti_white = 7
    invisible = 8


dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]
inf = 999999999


class ActionTranslator(object):

    def __init__(self):
        self.__20action, self.__20index = self._generate(20)
        self.__19action, self.__19index = self._generate(19)
        self.__10action, self.__10index = self._generate(10)
        self.__9action, self.__9index = self._generate(9)
        self.__actions = {20: self.__20action, 19: self.__19action, 10: self.__10action, 9: self.__9action}
        self.__indexes = {20: self.__20index, 19: self.__19index, 10: self.__10index, 9: self.__9index}

    def _generate(self, size):
        action = [torch.Tensor([[-1, -1, -1, -1, -1]])]
        index = []
        for _ in range(size + 1):
            index.append([])
            for __ in range(size + 1):
                index[_].append([])
                for ___ in range(4):
                    index[_][__].append([])
                    for ____ in range(2):
                        index[_][__][___].append([])
        index[0][0][0][0] = 0
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                for k in range(4):
                    tgx = i + dx[k]
                    tgy = j + dy[k]
                    if tgx < 1 or tgx > size or tgy < 1 or tgy > size:
                        continue
                    index[i][j][k][0] = len(action)
                    action.append(torch.Tensor([[i, j, tgx, tgy, 0]]))
                    index[i][j][k][1] = len(action)
                    action.append(torch.Tensor([[i, j, tgx, tgy, 1]]))

        return action, index

    def i_to_a(self, size: int, index: int) -> torch.Tensor:
        return self.__actions[size][index]

    def a_to_i(self, size: int, action: torch.Tensor) -> int:
        direction = 0
        for i in range(4):
            if action[0][2] == action[0][0] + dx[i] and action[0][3] == action[0][1] + dy[i]:
                direction = i
        return self.__indexes[size][action[0][0]][action[0][1]][direction][action[0][4]]

    def mask(self, obs, map_size):
        o = obs[0]
        mask_vec = torch.zeros([len(self.__actions[map_size])], dtype=torch.long)
        for _a in range(len(self.__actions[map_size])):
            act = self.__actions[map_size][_a][0]
            act -= 1
            act[4] += 1

            if int(o[2][int(act[1]) - 1][int(act[0]) - 1]) != 0:
                # 不是自己的
                mask_vec[_a] = -inf
            else:
                mask_vec[_a] = 1
        return mask_vec.to(device)


at = ActionTranslator()
device = torch.device("cuda")