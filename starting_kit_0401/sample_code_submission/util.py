import os

# import psutil

import pickle
import time
import signal
import math
from contextlib import contextmanager

from typing import Any

import CONSTANT

nesting_level = 0
is_start = None

class TimeoutException(Exception):
    pass

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.remain)
        start_time = time.time()
        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            remain_time = math.ceil(self.total - self.exec)
            self.remain = remain_time

            log(f'{pname} success, time spent so far {self.exec} sec')

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")

def df_memory_usage(X):
    return X.memory_usage().sum() # byte

''' #require psutil
def get_process_memory():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 # /1024 # MB
    except:
        return 0
'''

def get_process_memory():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip()) / 1024 #MB

def check_imbalance_data(y):
    if y.value_counts()[0] / len(y) < CONSTANT.IMBALANCE_RATE:
        return 0
    elif y.value_counts()[1] / len(y) < CONSTANT.IMBALANCE_RATE:
        return 1
    else:
        return None

class Config:
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        import numpy as np

        def my_nunique(x):
            return x.nunique()

        my_nunique.__name__ = 'nunique'
        ops = {
            CONSTANT.NUMERICAL_TYPE: ["mean", "sum", "min", "max", "std"],
            CONSTANT.CATEGORY_TYPE: ["count"],
            #  TIME_TYPE: ["max"],
            #  MULTI_CAT_TYPE: [my_unique]
        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]
        if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            assert False, f"MultiCategory type feature's aggregate op are not supported."
            return ops[CONSTANT.MULTI_CAT_TYPE]
        if col.startswith(CONSTANT.TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."
        assert False, f"Unknown col type {col}"

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)
