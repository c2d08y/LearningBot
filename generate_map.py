# python version of https://github.com/By-Ha/Checkmate/blob/master/game/map.js
import random
from const import *


def map_to_tensor(game_map):
    size = len(game_map) - 1
    t = torch.zeros([4, size, size])
    type_trans = [BlockType.road, BlockType.crown, BlockType.road, BlockType.city, BlockType.mountain, BlockType.city]
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            t[0][i - 1][j - 1] = game_map[i][j]["amount"]
            t[1][i - 1][j - 1] = type_trans[game_map[i][j]["type"]]
            t[2][i - 1][j - 1] = game_map[i][j]["color"]
            t[3][i - 1][j - 1] = 1
    return t


def generate_random_map(player):
    def rnd(num):
        t = round(random.random() * num)
        return num if t == 0 else t

    def a_star(x, y, tar_x, tar_y):
        vis = []
        q = []
        d = [[1, -1, 0, 0], [0, 0, 1, -1]]
        for i in range(size + 1):
            vis.append([])
            for j in range(size + 1):
                vis[i].append([])
        q.append([x, y, 0])
        vis[x][y] = 1
        while len(q) > 0:
            tx = q[0][0]
            ty = q[0][1]
            step = q[0][2]
            q = q[1::]
            for j in range(4):
                tx2 = tx + d[0][j]
                ty2 = ty + d[1][j]
                if tx2 > size or ty2 > size or tx2 <= 0 or ty2 <= 0 or gm[tx2][ty2]["type"] == 4 or vis[tx2][ty2]:
                    continue
                vis[tx2][ty2] = 1
                q.append([tx2, ty2, step + 1])
                if tx2 == tar_x and ty2 == tar_y:
                    return step + 1
        return -1

    gm = []
    size = 0
    if player == 2:
        size = 10
    else:
        size = 20
    for i in range(size + 1):
        gm.append([])
        for j in range(size + 1):
            gm[i].append({"color": 0, "type": 0, "amount": 0})
    gm[0][0] = {size: size}
    i = 1
    while i <= 0.13 * size * size:
        for tt in range(1, 11):
            rnd(size)
        t1 = rnd(size)
        t2 = rnd(size)
        while gm[t1][t2]["type"] != 0:
            t1 = rnd(size)
            t2 = rnd(size)
        gm[t1][t2]["type"] = 4
        i += 1
    i = 1
    while i <= 0.05 * size * size:
        for tt in range(1, 11):
            rnd(size)
        t1 = rnd(size)
        t2 = rnd(size)
        while gm[t1][t2]["type"] != 0:
            t1 = rnd(size)
            t2 = rnd(size)
        gm[t1][t2]["type"] = 5
        gm[t1][t2]["amount"] = int(rnd(10)) + 40
        i += 1
    last = []
    calcTimes = 0
    for i in range(1, player + 1):
        calcTimes += 1
        if calcTimes >= 100:
            return generate_random_map(player)
        t1 = rnd(size - 2) + 1
        t2 = rnd(size - 2) + 1
        # 至少留一个方位有空
        while gm[t1][t2]["type"] != 0 or (gm[t1 + 1][t2]["type"] != 0 and gm[t1 - 1][t2]["type"] != 0
                                          and gm[t1][t2 + 1]["type"] != 0 and gm[t1][t2 + 1]["type"] != 0):
            t1 = rnd(size - 2) + 1
            t2 = rnd(size - 2) + 1

        if i == 1:
            gm[t1][t2]["color"] = i
            gm[t1][t2]["amount"] = 1
            gm[t1][t2]["type"] = 1
        else:
            flag = 0
            for j in range(len(last)):
                if a_star(t1, t2, last[j][0], last[j][1]) > 6:
                    continue
                flag = 1
                i -= 1
                break
            if flag == 0:
                gm[t1][t2]["color"] = i
                gm[t1][t2]["amount"] = 1
                gm[t1][t2]["type"] = 1
        last.append([t1, t2])
    gm[0][0]["type"] = 1
    return gm
