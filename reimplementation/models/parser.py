#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys
from reimplementation.models.helpers.defaults import get_cfg

from os.path import exists

def load_config(cfg_file):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if exists(cfg_file):
        cfg.merge_from_file(cfg_file)
    else:
        print("Fail in loading configs")