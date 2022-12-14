import math
import os
import time
from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException, TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from const import *
from networks import *
from settings import *


class next_round(object):

    def __init__(self, old):
        self.old = old
        self.new = None

    def __call__(self, driver):
        try:
            new = driver.find_element(By.CSS_SELECTOR, ".own.crown").text
        except Exception:
            return False
        return new != self.old


def wait_until(check_func, *args, poll_frequency=0.01):
    """
    强制等待 不会超时
    :param check_func: 检测函数
    :param args: arguments
    :param poll_frequency: sleep interval between calls
    :return:
    """
    while True:
        if check_func(*args):
            break
        time.sleep(poll_frequency)


def print_with_color(style=0, front=0, background=0, content="", end='\n'):
    print(f"\033[{style};{front};{background}m{content}\033[0m", end=end)


def add_crown_and_city(game_map):
    for i in range(1, len(game_map)):
        for j in range(1, len(game_map)):
            if game_map[i][j]["type"] == 1 or game_map[i][j]["type"] == 3:
                game_map[i][j]["amount"] += 1
    return game_map


def add_road(game_map):
    for i in range(1, len(game_map)):
        for j in range(1, len(game_map)):
            if game_map[i][j]["type"] == 2 and game_map[i][j]["amount"] > 0 and game_map[i][j]["type"] != 0:
                game_map[i][j]["amount"] += 1
    return game_map


def combine(f, t, cnt, game_map):
    player_die = 0
    if t["color"] == f["color"]:
        t["amount"] += cnt
        f["amount"] -= cnt
    else:
        t["amount"] -= cnt
        f["amount"] -= cnt
        if t["amount"] < 0:
            if t["type"] == 1:
                tcolor = t["color"]
                for i in range(1, len(game_map)):
                    for j in range(1, len(game_map)):
                        if game_map[i][j]["color"] == tcolor:
                            game_map[i][j]["color"] = f["color"]
                            if game_map[i][j]["type"] == 1:
                                game_map[i][j]["type"] = 3
                                player_die += 1
            elif t["type"] == 5:
                t["type"] = 3
            elif t["type"] != 3:
                t["type"] = 2
            t["color"] = f["color"]
            t["amount"] = -t["amount"]
    return game_map, f, t, player_die


def sim_next_round(game_map, movement, rnd, players):
    game_map = add_crown_and_city(game_map)
    if rnd % 10 == 0:
        game_map = add_road(game_map)
    for i in movement:
        mv = movement[i]
        if not mv:
            continue
        f = game_map[mv[0]][mv[1]]
        t = game_map[mv[2]][mv[3]]
        if not f or f["color"] == 0 or f["color"] != players[i]["color"]:
            continue
        tmp = f["amount"]
        if mv[4] == 1:
            tmp = math.ceil((f["amount"] + 0.5) / 2)
        game_map, f, t, player_die = combine(f, t, tmp - 1, game_map)
        game_map[mv[0]][mv[1]] = f
        game_map[mv[2]][mv[3]] = t
    return game_map


def print_map(game_map):
    color_trans = [37, 34, 31, 32, 33, 36, 35, 30]
    type_trans = [0, 4, 0, 3, 7, 3]
    size = game_map[0][0]["size"]
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            bg = 47 if game_map[i][j]["type"] == 3 or game_map[i][j]["type"] == 5 else 48
            print_with_color(style=type_trans[game_map[i][j]["type"]],
                             front=color_trans[game_map[i][j]["color"]],
                             background=bg, content=game_map[i][j]["amount"], end='\t')
        print()


def print_tensor_map(game_map):
    color_trans = [0, 44, 41, 42, 43, 46, 45, 105, 103]
    type_trans = {
        BlockType.road: Style.default,
        BlockType.crown: Style.under_line,
        BlockType.city: Style.italic,
        BlockType.mountain: Style.default
    }
    size = game_map.shape[1]
    for i in range(size):
        for j in range(size):
            if game_map[2][i][j] == PlayerColor.grey and \
                    game_map[1][i][j] == BlockType.city or game_map[1][i][j] == BlockType.mountain:
                bg = 40
            else:
                bg = color_trans[int(game_map[2][i][j])]
            ctt = '#' if game_map[1][i][j] == BlockType.mountain else str(int(game_map[0][i][j]))
            print_with_color(style=type_trans[int(game_map[1][i][j])],
                             front=97,
                             background=bg, content=ctt, end='\t')
        print()


def login(driver):
    """
    登录
    需要手动输入验证码
    :return:
    """
    driver.get("https://kana.byha.top:444/login")
    driver.find_element_by_name("username").send_keys(bot_name)
    driver.find_element_by_name("pwd").send_keys(bot_pwd)
    while True:
        try:
            time.sleep(3)
            driver.find_element_by_id("submitButton")
            driver.execute_script("alert('帮忙输一下验证码好不好qaq 等你10秒哦~~')")
            time.sleep(5)
            try:
                driver.switch_to.alert.accept()
                driver.switch_to.window(driver.window_handles[0])
            except NoAlertPresentException:
                pass
            time.sleep(5)
            driver.find_element_by_id("submitButton").click()
        except NoSuchElementException:
            break

    try:
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.ID, "JoinRoom")))
    except TimeoutException:
        driver.quit()
        raise "room cannot enter"


def init_driver_options() -> webdriver.ChromeOptions:
    """
    浏览器的设定
    :return: options
    """
    options = webdriver.ChromeOptions()
    prefs = {"credentials_enable_service": False, "profile.password_manager_enabled": False,
             "profile.default_content_setting_values.notifications": 2}
    prefs.update({'download.prompt_for_download': False,
                  'download.default_directory': r'D:\MyFiles\LearningBot\records'})
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    return options


def get_model(model_name: str, model_path, map_size):
    if os.path.exists(model_path):
        return torch.load(model_path)
    else:
        return Actor(map_size) if model_name.lower() == "actor" else Critic(map_size)


def map_to_tensor(game_map):
    size = len(game_map) - 1
    t = torch.zeros([4, size, size])
    type_trans = [BlockType.road, BlockType.crown, BlockType.road, BlockType.city, BlockType.mountain, BlockType.city]
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            t[0][i - 1][j - 1] = game_map[i][j]["amount"]
            t[1][i - 1][j - 1] = type_trans[int(game_map[i][j]["type"])]
            t[2][i - 1][j - 1] = game_map[i][j]["color"]
            t[3][i - 1][j - 1] = 1
    return t