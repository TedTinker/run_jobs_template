
import gzip
import os
import pickle
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from utils import folder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def to_plain(obj):
    """A copy of a training log with tensors turned into floats, so the pickle
    loads without torch and every series is plain numbers."""
    if isinstance(obj, dict):
        return {k : to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, 'item') and getattr(obj, 'numel', lambda: 1)() == 1:
        return obj.item()
    return obj


def for_buffer(value_dict):
    return {k : v.squeeze(0).squeeze(0) for k, v in value_dict.items()}


def close_buffer_episode(buffer, next_observation_dict):
    """The buffer only closes an episode on done or at max_steps. If an episode is
    ever cut short, close it here, storing the last real observation, so the next
    episode does not write into the same slot. A no-op when the buffer already closed it."""
    if buffer.current_slot is None:
        return
    for k, v in next_observation_dict.items():
        buffer.observation_buffers[k].push(buffer.current_slot, buffer.time_ptr, v)
    buffer.commit_episode()



class Agent:

    def __init__(self, args, i):
        self.args = args
        self.i = i



    #------------------
    # Called by main.py.
    #------------------

    def training(self, q):
        args = self.args
        q.put((self.i, 1.))