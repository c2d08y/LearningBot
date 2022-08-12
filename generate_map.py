import execjs
from utils import *


with open("generate_map.js", "r", encoding="utf-8") as _js_file:
    _js_code = _js_file.read()

_generator = execjs.compile(_js_code)


def generate_random_map(player):
    return map_to_tensor(_generator.call("generateRandomMap", player))


def generate_maze_map(player):
    return map_to_tensor(_generator.call("generateMazeMap", player))


def generate_empty_map(player):
    return map_to_tensor(_generator.call("generateEmptyMap", player))