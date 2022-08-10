# python version of https://github.com/By-Ha/Checkmate/blob/master/game/map.js
import random
import functools
import settings
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


def generate_maze_map(player):
    def rnd(num):
        t = round(random.random() * num)
        return num if t == 0 else t

    gm = []
    size = 0
    id = {}
    etot = 0
    edges = []
    vtot = 0
    venum = []
    if player == 2:
        size = 9
    else:
        size = 19
    for i in range(size + 1):
        gm.append([])
        venum.append([])
        for j in range(size + 1):
            gm[i].append({"color": 0, "type": 0, "amount": 0})
            venum[i].append([])
    gm[0][0] = {size: size}
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if i % 2 == 0 and j % 2 == 0:
                gm[i][j]["type"] = 4
            if i % 2 == 1 and j % 2 == 1:
                venum[i][j] = vtot
                vtot += 1
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            tmp1 = i - 1
            tmp3 = j - 1
            tmp4 = j + 1
            tmp2 = i + 1

            if i % 2 == 0 and j % 2 == 1:
                venum[i][j] = etot
                edges.append({"a": venum[tmp1][j], "b": venum[tmp2][j], "w": 10 + int(rnd(10)), "posa": i, "posb": j})
                etot += 1
            if i % 2 == 1 and j % 2 == 0:
                venum[i][j] = etot
                edges.append({"a": venum[i][tmp3], "b": venum[i][tmp4], "w": 10 + int(rnd(10)), "posa": i, "posb": j})
                etot += 1

    def cmp(x, y):
        return x["w"] - y["w"]

    def find(x):
        if x == id[x]:
            return x
        id[x] = find(id[x])
        return id[x]

    edges.sort(key=functools.cmp_to_key(cmp))
    for i in range(vtot):
        id[i] = i
    for i in range(etot):
        if find(edges[i]["a"]) != find(edges[i]["b"]):
            id[find(edges[i]["a"])] = id[(edges[i]["b"])]
            gm[edges[i]["posa"]][edges[i]["posb"]]["type"] = 5
            gm[edges[i]["posa"]][edges[i]["posb"]]["amount"] = 10
        else:
            gm[edges[i]["posa"]][edges[i]["posb"]]["type"] = 4
    calcTimes = 0
    for i in range(1, player + 1):
        calcTimes += 1
        if calcTimes >= 100:
            return generate_maze_map(player)
        t1 = rnd(size)
        t2 = rnd(size)
        while True:
            t1 = rnd(size)
            t2 = rnd(size)
            tmpcnt = 0
            if t1 - 1 >= 1:
                if gm[t1 - 1][t2]["type"] != 4:
                    tmpcnt += 1
            if t2 - 1 >= 1:
                if gm[t1][t2 - 1]["type"] != 4:
                    tmpcnt += 1
            if t1 + 1 <= size:
                if gm[t1 + 1][t2]["type"] != 4:
                    tmpcnt += 1
            if t2 + 1 <= size:
                if gm[t1][t2 + 1]["type"] != 4:
                    tmpcnt += 1
            if gm[t1][t2]["type"] == 0 and tmpcnt == 1:
                break
        gm[t1][t2]["color"] = i
        gm[t1][t2]["amount"] = 1
        gm[t1][t2]["type"] = 1
    i = 1
    while i <= (size * size) / 15:
        tryTime = 0
        while True:
            tryTime += 1
            x = rnd(size)
            y = rnd(size)
            if tryTime >= 20:
                break
            flag = 0
            flagUD = 0
            flagLR = 0
            for t1 in range(-1, 2):
                for t2 in range(-1, 2):
                    if t1 == 0 and t2 == 0:
                        continue
                    if 0 < x + t1 <= size and y + t2 <= size:
                        if gm[x + t1][y + t2]["type"] == 1:
                            flag = 1
                            break
                if flag:
                    break
            if flag or x % 2 == y % 2:
                continue
            if gm[x][y]["type"] == 4:
                gm[x][y]["type"] = 5
                gm[x][y]["amount"] = 10
                break
        i += 1
    gm[0][0]["type"] = 1
    return gm


# if __name__ == '__main__':
#     if settings.debug:
#         from utils import *
#         print_tensor_map(map_to_tensor(generate_random_map(2)))
