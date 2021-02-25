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

from base import BasicPruning
from model_writers import Writer
from dependencies import SEResidualDependencyBlock
from weight_transfer import ResNetWeightTransferUnit

def nn_conv(layer, filterNum, opFilter):
#{{{
    if opFilter:
        layer.wShape[0] -= 1
        layer.wMask[filterNum] = 0
        if layer.bias is not None:
            layer.bShape[0] -= 1
            layer.bMask[filterNum] = 0
    else:
        layer.wShape[1] -= 1
        layer.wMask[:,filterNum,:,:] = 0
#}}}

def nn_linear(layer, filterNum, spatialDims):
#{{{
    layer.wShape[1] -= spatialDims
    idx = [filterNum+i for i in range(spatialDims)]
    layer.wMask[:,idx] = 0
#}}}

def se_module(module, filterNum):
#{{{
    convCount = 0
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Conv2d):
            assert convCount < 2, "SE Module has more than 2 convs"
            opF = False if convCount==0 else True
            nn_conv(mod, filterNum, opFilter=opF)
            convCount += 1
#}}}

def conv_layer(pruner, layerName, filterNum, findBN=True):
#{{{
    currLayer = pruner.allModules[layerName]
    nextLayerNames = pruner.depBlock.linkedConvAndFc[layerName]
    nextLayers = [pruner.allModules[x[0]] for x in nextLayerNames]

    # remove output filter for current layer
    nn_conv(currLayer, filterNum, opFilter=True)

    # remove input channel from next layers
    for nextLayerName, nextLayer in zip(nextLayerNames, nextLayers):
        if isinstance(nextLayer, nn.Conv2d):
            nn_conv(nextLayer, filterNum, opFilter=False)
        elif isinstance(nextLayer, nn.Linear):
            #NOTE: parallelLayers only really needed for GoogleNet as there's a concatenation
            parallelLayerNames = [k for k,v in pruner.depBlock.linkedConvAndFc.items()\
                    if v[0][0] == nextLayerName[0]]
            #NOTE:totalOpChannels only really needed for AlexNet as the first FC layer has a 6x6 
            #spatial input dimension to the FC, all other nets have 1x1
            totalOpChannels = sum([pruner.allModules[x].out_channels for x in parallelLayerNames])
            spatialDims = int(nextLayer.in_features / totalOpChannels)
            nn_linear(nextLayer, filterNum, spatialDims)
#}}}

def se_bottleneck(pruner, layerName, modName, mod, filterNum):
#{{{
    conv_layer(pruner, layerName, filterNum)
    modDetails = pruner.depBlock.moduleDetails[type(mod)]
    
    # check if module has downsampling conv and output conv of module is being pruned
    if modDetails['convs'][-1] in layerName:
        depCalc = pruner.depBlock.depCalcs[type(mod)]
        hasDS, dsName, dsMod = depCalc.check_for_downsample(mod, modDetails)
        if hasDS:
            nn_conv(dsMod, filterNum, opFilter=True)
            pruner.channelsToPrune[f"{modName}.{dsName}"] = pruner.channelsToPrune[layerName]

    # check if pruned conv has SE output
    for convName, seModName in modDetails['se'].items():
        if convName in layerName: 
            seMod = pruner.allModules[f"{modName}.{seModName}"] 
            se_module(seMod, filterNum)
#}}}

class SEResNetPruning(BasicPruning):
    def __init__(self, params, model):  
    #{{{
        logging.info("Initialising SEResNet Pruning")
        if params.seNet is not None and params.seNet['task'] == 'prune_and_retrain': 
            self.fileName = '{}_{}.py'.format(params.seNet['network'], params.pruner['pruning_perc'])
            self.netName = f"{params.seNet['network'].capitalize()}"
        else:
            raise ValueError("Could not fine se_net section in params file")
        
        self.conv_or_fc = lambda m : (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear))
        
        depBlock = SEResidualDependencyBlock(model)

        super().__init__(params, model, depBlock=depBlock)
    #}}}

    def inc_prune_rate_old(self, layerName, dependencies, updateGlobalParams=True):
    #{{{
        #TODO: Is there a better way to do this?
        
        lParam = str(layerName)
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1] * currLayerSize[2] * currLayerSize[3]
        # remove 1 output filter from current layer
        self.layerSizes[lParam][0] -= 1 
        
        # check if it is at the head of a dependency group, i.e. it has a downsampling layer
        if any(x.index(layerName) == 0 for x in dependencies if layerName in x):
            blockName = [_blockName for _, _blockName in self.depBlock.linkedModules\
                    if f"{_blockName}." in layerName][0] 
            module = [m for n,m in self.model.named_modules() if n == blockName][0]
            instances = [isinstance(module, x) for x in self.depBlock.instances]
            if True in instances: 
                instanceIdx = instances.index(True)
                dsInstName = self.depBlock.dsLayers[instanceIdx][0]
                dsLayer = [x for x,p in module.named_modules() if dsInstName in x and\
                        isinstance(p, torch.nn.Conv2d)]
                if len(dsLayer) != 0: 
                    dsLayer = dsLayer[0]
                    dsLayerName = f"{blockName}.{dsLayer}" 
                    self.layerSizes[dsLayerName][0] -= 1
                    dsLayerSize = self.layerSizes[dsLayerName]
                    paramsPruned += dsLayerSize[1] * dsLayerSize[2] * dsLayerSize[3]

        nextLayerDetails = self.depBlock.linkedConvAndFc[lParam]
        for (nextLayer, groups) in nextLayerDetails:
            nextLayerSize = self.layerSizes[nextLayer]

            # check if FC layer
            if len(nextLayerSize) == 2: 
                finalLayers = [k for k,v in self.depBlock.linkedConvAndFc.items() if v[0][0] == nextLayer]
                currOFMSize = sum([self.layerSizes[x][0] for x in finalLayers]) 
                fcParamsPruned = int(nextLayerSize[1] / (currOFMSize + 1))
                breakpoint()
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
            
        if updateGlobalParams:
            self.currParams -= paramsPruned
        
        return (100. * (1. - (self.currParams / self.totalParams)))
    #}}}
    
    def inc_prune_rate(self, layerName, filterNum):
    #{{{
        # currModName, _, currMod = list(filter(lambda x: f"{x[0]}." in layerName,\
        #         self.depBlock.linkedModulesWithFc))[0]
        currModName, _, currMod = list(filter(lambda x: f"{x[0]}" in layerName,\
                self.depBlock.linkedModulesWithFc))[0]
        if type(currMod) in self.depBlock.instances:
            # special block to deal with
            se_bottleneck(self, layerName, currModName, currMod, filterNum)
        else:
            conv_layer(self, layerName, filterNum)
        
        currParams = sum([torch.prod(m.wShape) for n,m in self.model.named_modules() if self.conv_or_fc(m)])
        return 100. * (1. - (float(currParams)/self.totalParams))
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
    
    def prune_model(self, model, transferWeights=True, pruneNum=None):
    #{{{
        # pruning based on l1 norm of weights
        if self.params.pruner['mode'] == 'l1-norm':
            logging.info(f"Pruning filters: l1-norm - Pruning Level {self.params.pruner['pruning_perc']}")
            self.calculate_metric = self.l1_norm
            channelsPruned = self.structured_l1_weight(model)
            newModelParams = sum([np.prod(p.shape) for p in self.model.parameters()])
            pruneRate = 100. * (1. - (newModelParams/self.totalParams))
            logging.info((f"Pruned Percentage = {pruneRate:.2f}%, New Model Size = {newModelParams*4/1e6:.2f} MB, "
                f"Orig Model Size = {self.totalParams*4/1e6:.2f} MB"))
            return channelsPruned
     #}}}
    
    def structured_l1_weight(self, model):
    #{{{
        localRanking, globalRanking = self.rank_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()
        
        pl = float(self.params.pruner['pruning_perc'])/100.
        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        if ('se_resnet' in self.params.arch): 
            minChannelsKept = [int(math.ceil(gs * 0.10)) for gs in numChannelsInDependencyGroup]
        else:
            ### This strategy ensures that at as many pruning levels as possible we maintain wider networks  
            ### At very high pruning levels (those in the if case above), we can't apply the same strategy
            ### as it limits the total amount of pruning possible (memory reduction) --> this is capped at 
            ### at most pruning 50% of the layer which allows desired memory reduction with some width maintenance
            ### This is enforced only for external dependencies, but internally within a block the internal layers
            ### can be pruned heavily
            if pl <= 0.5: 
                minChannelsKept = [int(math.ceil(gs * (1.0 - pl))) for gs in numChannelsInDependencyGroup]
            else:
                minChannelsKept = [int(math.ceil(gs * pl)) for gs in numChannelsInDependencyGroup]
        
        minChannelsKept = [self.minFiltersInLayer for k in internalDeps] + minChannelsKept
        
        dependencies = internalDeps + externalDeps
        self.remove_filters(localRanking, globalRanking, dependencies, minChannelsKept)

        return self.channelsToPrune
    #}}}
    
    def l1_norm(self, conv, module):
    #{{{
        layer = dict(self.model.named_modules())[conv]
        param = layer.weight.data.numpy()
        metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
        metric /= (param.shape[1]*param.shape[2]*param.shape[3])
        return metric
    #}}}
    
    def rank_filters(self, model):
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
    
    def remove_filters(self, localRanking, globalRanking, dependencies, minChannelsKept):
    #{{{
        [m.register_buffer('wMask', torch.ones_like(m.weight), persistent=False)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m)]
        [m.register_buffer('wShape', torch.tensor(m.weight.shape),persistent=False)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m)]
        [m.register_buffer('bMask', torch.ones_like(m.bias), persistent=False)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m) and m.bias is not None]
        [m.register_buffer('bShape', torch.tensor(m.bias.shape), persistent=False)\
                for n,m in self.model.named_modules() if self.conv_or_fc(m) and m.bias is not None]
        
        listIdx = 0
        currentPruneRate = 0
        self.currParams = self.totalParams
        self.allModules = dict(self.model.named_modules())
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
                print((f"Pruning: {end-start:.6f}s : {listIdx:05d}/{len(globalRanking)}, "
                    f"{currentPruneRate:.4f}, {float(self.params.pruner['pruning_perc']):.1f}"),\
                            end='\r', flush=True)
            
            listIdx += 1
        
        print()
        self.start = time.time()
        self.construct_pruned_model()
        return self.channelsToPrune
    #}}}
    
    def write_net(self, printFileLoc=True):
    #{{{
        if printFileLoc:
            print("Pruned model written to {}".format(self.filePath))
        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        
        self.writer = Writer(self.netName, channelsPruned, self.depBlock, self.filePath, self.layerSizes)
        
        lTypes, lNames = zip(*self.depBlock.linkedModules)
        prunedModel = copy.deepcopy(self.model)
        for n,m in prunedModel.named_modules(): 
            # detect dependent modules and convs
            if any(n == x for x in lNames):
                idx = lNames.index(n) 
                lType = lTypes[idx]
                self.writer.write_module(lType, n, m)
            
            # ignore recursion into dependent modules
            elif any(f'{x}.' in n for t,x in self.depBlock.linkedModules):
                continue

            # all other modules in the network
            else:
                try: 
                    if not any(f"{blockName}." in n for blockName in self.blocksToRemove):
                        self.writer.write_module(type(m).__name__.lower(), n, m)
                except KeyError:
                    if self.verbose:
                        print("CRITICAL WARNING : layer found ({}) that is not handled in writers. This could potentially break the network.".format(type(m)))
        
        self.writer.write_network()       
    #}}}
    
    def transfer_weights(self, oModel, pModel): 
    #{{{
        lTypes, lNames = zip(*self.depBlock.linkedModules)
        
        pModStateDict = pModel.state_dict() 

        self.wtu = ResNetWeightTransferUnit(self, pModStateDict, self.channelsToPrune, self.depBlock,\
                self.layerSizes)
        
        mutableOModel = copy.deepcopy(oModel)
        for n,m in mutableOModel.named_modules(): 
            [x.detach_() for x in m.parameters()]
            # detect dependent modules and convs
            if any(n == x for x in lNames):
                idx = lNames.index(n) 
                lType = lTypes[idx]
                self.wtu.transfer_weights(lType, n, m)
            
            # ignore recursion into dependent modules
            elif any(f'{x}.' in n for t,x in self.depBlock.linkedModules):
                continue
            
            # all other modules in the network
            else:
                try: 
                    if not any(f"{blockName}." in n for blockName in self.blocksToRemove):
                        self.wtu.transfer_weights(type(m).__name__.lower(), n, m)
                except KeyError as e:
                    if self.verbose:
                        print(f"CRITICAL WARNING : layer found ({type(m)}) that is not handled in weight transfer. This could potentially break the network.")
        
        pModel.load_state_dict(pModStateDict)
        return pModel 
    #}}}

