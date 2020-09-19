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

class GoogLeNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
    #{{{
        self.fileName = 'googlenet_{}.py'.format(int(params.pruner['pruning_perc']))
        self.netName = 'GoogLeNet'
        
        super().__init__(params, model)
    #}}}
    
    def inc_prune_rate(self, layerName):
    #{{{
        lParam = str(layerName)
        # remove 1 output filter from current layer
        self.layerSizes[lParam][0] -= 1 
        
        nextLayerDetails = self.depBlock.linkedConvAndFc[lParam]
        for (nextLayer, groups) in nextLayerDetails:
            nextLayerSize = self.layerSizes[nextLayer]
            currLayerSize = self.layerSizes[lParam]
            paramsPruned = currLayerSize[1]*currLayerSize[2]*currLayerSize[3]

            # check if FC layer
            if len(nextLayerSize) == 2: 
                finalLayers = [k for k,v in self.depBlock.linkedConvAndFc.items() if v[0][0] == nextLayer]
                currOFMSize = sum([self.layerSizes[x][0] for x in finalLayers]) 
                # fcParamsPruned = int(nextLayerSize[1] / (currLayerSize[0] + 1))
                fcParamsPruned = int(nextLayerSize[1] / (currOFMSize + 1))
                paramsPruned += fcParamsPruned * nextLayerSize[0]
                self.layerSizes[nextLayer][1] -= fcParamsPruned
            else:
                #TODO:
                # this is assuming we only have either non-grouped convolutions or dw convolutions
                # have to change to accomodate grouped convolutions
                nextOpChannels = nextLayerSize[0] if groups == 1 else 1
                paramsPruned += nextOpChannels*nextLayerSize[2]*nextLayerSize[3]
                
                # remove 1 input activation from next layer if it is not a dw conv
                if groups == 1:
                    self.layerSizes[nextLayer][1] -= 1
            
        self.currParams -= paramsPruned
        
        return (100. * (1. - (self.currParams / self.totalParams)))
    #}}}
#}}}
