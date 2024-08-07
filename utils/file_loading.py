import json


def load_json(file_path):
    with open(file_path, "r") as file:
        f = json.load(file)
    return f
