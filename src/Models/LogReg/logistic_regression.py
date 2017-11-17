from __future__ import division
import sys
import json
import math
import os
import numpy as np
from collections import defaultdict
import operator


def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.

    w2vec = {}
    with open(filename, "r") as f_in:
        for line in f_in:
            line_split = line.replace("\n", "").split()
            w = line_split[0]
            vec = np.array([float(x) for x in line_split[1:]])
            w2vec[w] = vec
    return w2vec
