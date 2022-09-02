#!/usr/bin/env python3
# Copyright (c) Facebook. and its affiliates. All Rights Reserved

import contextlib
import shutil
import tempfile
import os, torch, os.path as osp
import time

@contextlib.contextmanager
def make_temp_directory(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

def make_saved_dir(save : str):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    dateTime = time.localtime(int(time.time()))
    otherStyleTime = time.strftime("%m_%d_%Y_%H_%M_%S", dateTime)
    if not os.path.exists(os.path.join(OUTPUT_DIR, save)):
        try:
            save_ = os.path.join(OUTPUT_DIR, save + otherStyleTime)
            os.makedirs(save_)
        except:
            raise Exception("Created Saved Dir Failed!")
    else: print("Saved dir already exist!")
    return save_

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def save_checkpoint(state, filename, logger):
    if os.path.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(
                "Find {:} exist, delete is at first before saving".format(filename)
            )
        os.remove(filename)
    torch.save(state, filename)
    assert os.path.isfile(
        filename
    ), "save filename : {:} failed, which is not found.".format(filename)
    return filename


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "core/config")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "res")
