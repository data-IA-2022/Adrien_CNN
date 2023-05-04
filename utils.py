import time
from os.path import dirname, join, getsize


def relative_path(*path):

    return join(dirname(__file__), *path)


def calc_time(start_time):

    d = time.time() - start_time
    h = int(d / 3600)
    h = f"{h} h " if d > 3600 else ''
    m = int(d % 3600 / 60)
    m = f"{m:02} m " if d >= 3600 else f"{m} m " if d > 60 else ''
    s = int(d % 3600 % 60)
    s = f"{s:02} s" if d >= 60 else f"{s} s"
    return h + m + s


def get_size(file_path, unit='auto', txt=True, round_=3):

    file_size = getsize(file_path)
    exponents_map = {'GB': 3, 'MB': 2, 'KB': 1, 'B': 0}

    if file_size == 0:
        unit = 'B'

    if unit not in exponents_map:
        for key in exponents_map:
            if file_size >= 1024 ** exponents_map[key]:
                unit = key
                break

    size = file_size / 1024 ** exponents_map[unit]

    if txt:
        return f"{round(size, round_)} {unit}"

    else:
        return round(size, round_)
