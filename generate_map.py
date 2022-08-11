import execjs


with open("generate_map.js", "r", encoding="utf-8") as _js_file:
    _js_code = _js_file.read()

_generator = execjs.compile(_js_code)


def generate_random_map(player):
    _generator.call("generateRandomMap", player)


def generate_maze_map(player):
    _generator.call("generateMazeMap", player)


def generate_empty_map(player):
    _generator.call("generateEmptyMap", player)