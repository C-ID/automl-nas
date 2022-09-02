#!/usr/bin/env python3

from functools import lru_cache
import random, torch, numpy as np
from iopath.common.file_io import PathManager
from functools import wraps

@lru_cache()
def get_path_manager() -> PathManager:
    """target:
    "//mobile-vision/mobile_cv/common:fb",
    """
    path_manager = PathManager()
    return path_manager


def wrap_seed(f):
    rand_seed = random.randint
    @wraps(f)
    def wrap_func(*args, **kwargs):
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        return f(*args, **kwargs)
    return wrap_func
