import sys
import csv
import os
import numpy as np
import time
from tqdm import tqdm
import json
import pickle
import subprocess
import importlib
import math
import copy

from base import BasicPruning

import torch
import torch.nn as nn

class VGGPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
    #{{{
        self.fileName = 'vgg_{}.py'.format(int(params.pruner['pruning_perc']))
        self.netName = 'VGG'
        
        super().__init__(params, model)
    #}}} 
#}}}
