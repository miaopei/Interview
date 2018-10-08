import json
import ujson

def monkey_path_json():
    json.__name__ = 'ujson'
    json.dumps = ujson.dumps
    json.loads = ujson.loads

monkey_path_json()

print("main.py", json.__name__)

import sub

