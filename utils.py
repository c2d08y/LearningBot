import math
import time
from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException, TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from settings import *


class next_round(object):

    def __init__(self, locator, old):
        self.locator = locator
        self.old = old

    def __call__(self, driver):
        try:
            new = driver.find_element(self.locator[0], self.locator[1])
        except Exception:
            return False
        return new != self.old


def wait_until(check_func, *args, poll_frequency=0.01):
    """
    force wait, no time out
    :param check_func: function to test if it's ok
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
            print_with_color(style=type_trans[game_map[i][j]["type"]],
                             front=color_trans[game_map[i][j]["color"]],
                             background=48, content=game_map[i][j]["amount"], end='\t')
        print()


def login(driver):
    """
    just as the name
    captcha input required
    :return: is successful
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
    just as the name
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