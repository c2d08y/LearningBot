import base64
import datetime
import json
import re
from time import sleep
import game


def at_player_by_uid(uid):
    return '[at,uid=' + str(uid) + ']'


class Bot(object):

    def __init__(self):
        self.kanaLink = 'https://kana.byha.top:444/'

        config = json.load(open("config.json", 'r'))
        self.username = config['username']  # 用户名
        self.password = config['password']  # 密码
        self.room_id = config['roomID']  # 房间号
        self.secretId = config['secretId']
        self.secretKey = config['secretKey']

        self.default_user_remain_win_time = 10
        self.last_update_time = 0

        # 以下是每日更新的数据
        self.user_remain_win_time = {}  # 每个玩家的单挑剩余次数
        self.game_count = []  # 每种对局的次数
        self.user_score = {}  # 每个玩家的分数


    def main(self):
        self.game = game.Game(self.driver)
        self.room.id = self.room_id
        self.on = True
        flag = False
        while True:
            try:
                if self.driver.find_element_by_id("game-status").get_attribute('innerHTML') != "游戏中":
                    if flag:
                        flag = False
                    sleep(0.2)
                    self.game.is_pre = False
                else:
                    self.room.free_time = 0
                    self.game.bot_move()
                    continue
                flag = True
            except:
                continue