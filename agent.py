
import gzip
import os
import pickle
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from utils import folder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



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