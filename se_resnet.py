import os
import sys
import csv
import time
import math
import copy
import json
import pickle
import logging
import importlib
import subprocess
from tqdm import tqdm
from functools import reduce

import numpy as np

import torch
import torch.nn as nn

import src.adapt.lego.utils as utils

from modules import *
from base import BasicPruning
from model_writers import Writer
from weight_transfer import ResNetWeightTransferUnit
from dependencies import SEResidualDependencyBlock

class SEResNetPruning(BasicPruning):
    def __init__(self, params, model):  
    #{{{
        logging.info("Initialising SEResNet Pruning")
        if params.seNet is not None: 
            self.fileName = '{}_{}.py'.format(params.seNet['network'], params.pruner['pruning_perc'])
            self.netName = f"{params.seNet['network'].capitalize()}"
        else:
            raise ValueError("Could not fine se_net section in params file")
        
        self.conv_or_fc = lambda m : (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear))
        
        depBlock = SEResidualDependencyBlock(model)

        super().__init__(params, model, depBlock=depBlock)
    #}}}

    def prune_model(self, inplace=False, scales=None):
    #{{{
        if self.params.pruner['mode'] == 'l1-norm':
            logging.info(f"Pruning filters: l1-norm - Pruning Level {self.params.pruner['pruning_perc']}")
            self.calculate_metric = self.l1_norm
        
        elif self.params.pruner['mode'] == 'se_scales':
            logging.info(f"Pruning filters: se_scales - Pruning Level {self.params.pruner['pruning_perc']}")
            assert scales is not None
            self.seScales = scales
            self.calculate_metric = self.se_scales

        else:
            raise ValueError(f"Unrecognised mode {self.params.pruner['mode']}")
            
        # self.model = self.model if inplace else copy.deepcopy(self.model)
        channelsPruned = self.structured_prune()
        newModelParams = sum([np.prod(p.shape) for p in self.model.parameters()])
        pruneRate = 100. * (1. - (newModelParams/self.totalParams))
        logging.info((f"Pruned Percentage = {pruneRate:.2f}%, "
                      f"New Model Size = {newModelParams*4/1e6:.2f} MB, "
                      f"Orig Model Size = {self.totalParams*4/1e6:.2f} MB"))
        
        return channelsPruned, self.model
     #}}}
    
    def l1_norm(self, conv, module):
    #{{{
        layer = dict(self.model.named_modules())[conv]
        param = layer.weight.data.numpy()
        metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
        metric /= (param.shape[1]*param.shape[2]*param.shape[3])
        return metric
    #}}}
    
    def se_scales(self, conv, module):
    #{{{
        seDetails = self.depBlock.moduleDetails[type(module)]['se']
        for k,v in seDetails.items(): 
            if k in conv:
                moduleName = conv.split(k)[0]
                scales = self.seScales[f"{moduleName}{v}"]
                break
        metric = scales.mean(axis=0)
        return metric
    #}}}
    
    def structured_prune(self):
    #{{{
        localRanking, globalRanking = self.rank_filters()
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()
        
        pl = float(self.params.pruner['pruning_perc'])/100.
        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        
        minChannelsKept = [int(math.ceil(gs * 0.10)) for gs in numChannelsInDependencyGroup]
        
        minChannelsKept = [self.minFiltersInLayer for k in internalDeps] + minChannelsKept
        
        dependencies = internalDeps + externalDeps
        self.remove_filters(localRanking, globalRanking, dependencies, minChannelsKept)

        return self.channelsToPrune
    #}}}
    
    def rank_filters(self):
    #{{{
        localRanking = {} 
        globalRanking = []
        for conv, module in self.depBlock.layersToPrune.items():
            metric = self.calculate_metric(conv, module)
            globalRanking += [(conv, filtNum, x) for filtNum,x in enumerate(metric)]
            localRanking[conv] = sorted([(filtNum, x) for filtNum,x in enumerate(metric)], key=lambda tup:tup[1])
        
        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
        self.channelsToPrune = {l:[] for l,_ in self.depBlock.layersToPrune.items()}

        return localRanking, globalRanking
    #}}}
    
    def inc_prune_rate(self, layerName, filterNum):
    #{{{
        layer_in_module = lambda x: (x[0] == layerName) or (f"{x[0]}." in layerName)
        currModName, _, currMod = list(filter(layer_in_module, self.depBlock.linkedModulesWithFc))[0]
        if type(currMod) in self.depBlock.instances:
            # special block to deal with
            se_residual_bottleneck(self, layerName, currModName, currMod, filterNum)
        else:
            conv_layer(self, layerName, filterNum)
        
        self.currParams = sum([torch.prod(m.wShape) for n,m in self.model.named_modules()\
                if self.conv_or_fc(m)])
        return 100. * (1. - (float(self.currParams)/float(self.prunableParams)))
    #}}}
    
    def remove_filters(self, localRanking, globalRanking, dependencies, minChannelsKept):
    #{{{
        [m.register_buffer('wMask', torch.ones_like(m.weight), persistent=True)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m)]
        [m.register_buffer('wShape', torch.tensor(m.weight.shape),persistent=True)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m)]
        [m.register_buffer('bMask', torch.ones_like(m.bias), persistent=True)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m) and m.bias is not None]
        [m.register_buffer('bShape', torch.tensor(m.bias.shape), persistent=True)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m) and m.bias is not None]
        
        listIdx = 0
        currentPruneRate = 0
        self.currParams = self.totalParams
        self.allModules = dict(self.model.named_modules())
        self.prunableParams = sum([torch.prod(m.wShape) for n,m in self.model.named_modules()\
                if self.conv_or_fc(m)])
        start= time.time()
        while (currentPruneRate < float(self.params.pruner['pruning_perc'])) and (listIdx < len(globalRanking)):
            layerName, filterNum, _ = globalRanking[listIdx]

            depLayers = []
            pruningLimit = self.minFiltersInLayer
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    pruningLimit = minChannelsKept[i]
                    break
            
            # if layer not in group, just remove filter from layer 
            # if layer is in a dependent group remove corresponding filter from each layer
            depLayers = [layerName] if depLayers == [] else depLayers
            for layerName in depLayers:
                # case where you want to skip layers
                # if layers are dependent, skipping one means skipping all the dependent layers
                if len(localRanking[layerName]) <= pruningLimit:
                    continue
            
                # if filter has already been pruned, continue
                # could happen to due to dependencies
                if filterNum in self.channelsToPrune[layerName]:
                    continue 
                
                localRanking[layerName].pop(0)
                self.channelsToPrune[layerName].append(filterNum)
                
                currentPruneRate = self.inc_prune_rate(layerName, filterNum) 
                end = time.time()
                pl = float(self.params.pruner['pruning_perc'])
                print((f"Pruning: {end-start:.6f}s : Filter {listIdx:05d}/{len(globalRanking)}, "
                       f"Predicted prune rate {currentPruneRate:.4f}/{pl:.1f}%"), end='\r', flush=True)
            
            listIdx += 1
        
        print()
        self.start = time.time()
        self.construct_pruned_model()
        return self.channelsToPrune
    #}}}

    def construct_pruned_model(self):
    #{{{
        for n,m in self.model.named_modules():
            if isinstance(m, nn.Conv2d): 
                if not torch.eq(torch.tensor(m.weight.shape), m.wShape).all():
                    print(f"Reconstructing: {time.time()-self.start:.6f}s", end='\r', flush=True)
                    m.in_channels = m.wShape[1]
                    m.out_channels = m.wShape[0]
                    m.weight = torch.nn.Parameter(m.weight[m.wMask.bool()].view(tuple(m.wShape)))
                    if m.bias is not None:
                        m.bias = torch.nn.Parameter(m.bias[m.bMask.bool()].view(tuple(m.bShape)))
                lastConvName = n
                lastConvShape = m.weight.shape
            
            elif isinstance(m, nn.BatchNorm2d):
                if not torch.eq(torch.tensor(m.weight.shape), lastConvShape[0]).all():
                    print(f"Reconstructing: {time.time()-self.start:.6f}s", end='\r', flush=True)
                    remChannels = [x for x in range(m.weight.shape[0])\
                            if x not in self.channelsToPrune[lastConvName]]
                    m.num_features = len(remChannels)
                    m.weight = torch.nn.Parameter(m.weight[remChannels])
                    m.bias = torch.nn.Parameter(m.bias[remChannels])
                    m.running_mean = m.running_mean[remChannels]
                    m.running_var = m.running_var[remChannels]
            
            elif isinstance(m, nn.Linear):
                if not torch.eq(torch.tensor(m.weight.shape), m.wShape).all():
                    print(f"Reconstructing: {time.time()-self.start:.6f}s", end='\r', flush=True)
                    m.in_features = m.wShape[1]
                    m.out_features = m.wShape[0]
                    m.weight = torch.nn.Parameter(m.weight[m.wMask.bool()].view(tuple(m.wShape)))
                    if m.bias is not None:
                        m.bias = torch.nn.Parameter(m.bias[m.bMask.bool()].view(tuple(m.bShape)))
        print()
    #}}}

