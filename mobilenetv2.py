import os
import sys
import csv
import time
import json
import math
import copy
import pickle
import importlib
import subprocess
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from base import BasicPruning

class MobileNetV2PruningDependency(BasicPruning):
    def __init__(self, params, model):
    #{{{
        self.fileName = 'mobilenetv2_{}.py'.format(int(params.pruner['pruning_perc']))
        self.netName = 'MobileNetV2'
        
        super().__init__(params, model)
    #}}}
