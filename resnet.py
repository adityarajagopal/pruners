import os
import sys
import csv
import time
import math
import copy
import json
import pickle
import importlib
import subprocess
from tqdm import tqdm
from functools import reduce

import numpy as np

import torch
import torch.nn as nn

import src.adapt.lego.utils as utils
from base import BasicPruning
from model_writers import Writer, GoogLeNetWriter
from weight_transfer import ResNetWeightTransferUnit

class ResNet20PruningDependency(BasicPruning):
    def __init__(self, params, model):  
    #{{{
        if params.ofa is not None and params.ofa['task'] == 'prune_and_retrain': 
            self.fileName = 'ofa_{}_{}.py'.format(params.ofa['network'], params.pruner['pruning_perc'])
            self.netName = f"OFA{params.ofa['network'].capitalize()}"
        else:
            self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruner['pruning_perc']))
            self.netName = 'ResNet{}'.format(int(params.depth))

        if eval(params.pruner['prune_layers']) == True: 
            self.fileName = f"{self.fileName.split('.')[0]}_layerprune.py"
        
        super().__init__(params, model)
    #}}}
    
    def inc_prune_rate(self, layerName, dependencies, updateGlobalParams=True):
    #{{{
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
    
    def inc_prune_rate_block(self, blockName, localRanking, dependencies):
    #{{{
        ### update prune rate
        for l,nl in self.depBlock.linkedConvAndFc.items(): 
            if f"{blockName}." in l:
                channelsLeftToPrune = len(localRanking[l])
                del localRanking[l]
                del self.channelsToPrune[l]
                [self.inc_prune_rate(l, dependencies, False) for i in range(channelsLeftToPrune)][-1]

        ### update layerSizes        
        module = [m for n,m in self.model.named_modules() if n == blockName][0]
        instanceIdx = [isinstance(module, x) for x in self.depBlock.instances].index(True)
        internalLayerNames = self.depBlock.convs[instanceIdx]
        dsLayerName = self.depBlock.dsLayers[instanceIdx][0]
            
        prevBlock = [self.depBlock.linkedModules[i-1][1] for i,x in enumerate(self.depBlock.linkedModules)\
                if x[1] == blockName][0]
        
        prevLayer = f"{prevBlock}.{internalLayerNames[2]}"
        currLayer = f"{blockName}.{internalLayerNames[2]}"
        currIpChannels = self.layerSizes[f"{blockName}.{internalLayerNames[0]}"][1]

        for nLayer,_ in self.depBlock.linkedConvAndFc[currLayer]:
            self.layerSizes[nLayer][1] = currIpChannels
        
        newLS = copy.deepcopy(self.layerSizes)
        for k,v in self.layerSizes.items(): 
            if f"{blockName}." in k: 
                del newLS[k]
        self.layerSizes = newLS

        ### update linked modules
        newLM = [x for x in self.depBlock.linkedModules if x[1] != blockName]
        self.depBlock.linkedModules = newLM
        
        ### update linkedConvAndFc and linkedConvs
        newLCF = {k:v for k,v in self.depBlock.linkedConvAndFc.items() if f"{blockName}." not in k}
        newLCF[prevLayer] = self.depBlock.linkedConvAndFc[currLayer]
        self.depBlock.linkedConvAndFc = newLCF
        self.depBlock.linkedConvs = self.depBlock.linkedConvAndFc

        ### updat pruned params and get current prune rate
        self.currParams = sum([np.prod(x) for _,x in self.layerSizes.items()])
        currentPruneRate = ((self.totalParams - self.currParams)/self.totalParams)*100.
        
        return currentPruneRate
    #}}}
    
    def structured_l1_weight_layer(self, model):
    #{{{
        localRanking, globalRanking = self.rank_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()
        
        pl = float(self.params.pruner['pruning_perc'])/100.
        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        
        groupHeadPruneLimit = (1.0-pl) if pl <= 0.5 else pl
        # groupPruningLimits = [int(math.ceil(0.3 * gs)) for gs in numChannelsInDependencyGroup]
        groupPruningLimits = [[int(math.ceil(groupHeadPruneLimit * gs)) if i==0\
                else int(math.ceil(0.3 * gs)) for i,_ in enumerate(externalDeps[j])]\
                for j,gs in enumerate(numChannelsInDependencyGroup)]
        groupPruningLimits = [self.minFiltersInLayer]*len(internalDeps) + groupPruningLimits
        
        dependencies = internalDeps + externalDeps
        self.remove_filters_and_layers(localRanking, globalRanking, dependencies, groupPruningLimits)

        return self.channelsToPrune
    #}}}
    
    def remove_filters(self, localRanking, globalRanking, dependencies, groupPruningLimits):
    #{{{
        listIdx = 0
        count0 = 0
        currentPruneRate = 0
        self.currParams = self.totalParams
        while (currentPruneRate < float(self.params.pruner['pruning_perc'])) and (listIdx < len(globalRanking)):
            layerName, filterNum, _ = globalRanking[listIdx]

            depLayers = []
            pruningLimit = self.minFiltersInLayer
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    pruningLimit = groupPruningLimits[i]
                    break
            
            # if layer not in group, just remove filter from layer 
            # if layer is in a dependent group remove corresponding filter from each layer
            depLayers = [layerName] if depLayers == [] else depLayers
            if hasattr(self.depBlock, 'ignore'): 
                netInst = type(self.model.module)
                ignoreLayers = any(x in self.depBlock.ignore[netInst] for x in depLayers)
            else:
                ignoreLayers = False
            if not ignoreLayers:
                for layerName in depLayers:
                    # case where you want to skip layers
                    # if layers are dependent, skipping one means skipping all the dependent layers
                    if len(localRanking[layerName]) <= pruningLimit:
                        count0 += 1
                        continue
               
                    # if filter has already been pruned, continue
                    # could happen to due to dependencies
                    if filterNum in self.channelsToPrune[layerName]:
                        continue 
                    
                    localRanking[layerName].pop(0)
                    self.channelsToPrune[layerName].append(filterNum)
                    
                    currentPruneRate = self.inc_prune_rate(layerName, dependencies) 
            
            listIdx += 1
        
        print(f"Number of times min filter count hit = {count0}")
        return self.channelsToPrune
    #}}}
    
    def remove_filters_and_layers(self, localRanking, globalRanking, dependencies, groupPruningLimits):
    #{{{
        def check_prune_rate(cpr): 
            if cpr < float(self.params.pruner['pruning_perc']): 
                return True
            return False
        
        listIdx = 0
        currentPruneRate = 0
        self.currParams = self.totalParams
        while check_prune_rate(currentPruneRate) and (listIdx < len(globalRanking)):
            layerName, filterNum, _ = globalRanking[listIdx]

            depLayers = []
            groupIdx = -1
            pruningLimit = self.minFiltersInLayer
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    groupIdx = group.index(layerName)
                    pruningLimit = groupPruningLimits[i][groupIdx]
                    break
            # if layer not in group, just remove filter from layer 
            # if layer is in a dependent group remove corresponding filter from each layer
            depLayers = [layerName] if depLayers == [] else depLayers
            
            for i,layerName in enumerate(depLayers):
                # case where you want to skip layers
                # if layers are dependent, skipping one means skipping all the dependent layers
                # i == 0 prevents us from removing head block from dependency group as it has stride 2 and
                # downsamples the activations --> necessary
                # len(depLayers) == 1 ensures that for a layer not in a dependency group block is still pruned
                if (layerName in localRanking.keys()):
                    if (len(localRanking[layerName]) <= pruningLimit):
                        blockName = [_blockName for _, _blockName in self.depBlock.linkedModules\
                                if f"{_blockName}." in layerName][0] 
                        if (i != 0 or len(depLayers) == 1) and blockName not in self.blocksToRemove:
                            self.blocksToRemove.append(blockName)
                            currentPruneRate = self.inc_prune_rate_block(blockName, localRanking,\
                                    dependencies)
                            if not check_prune_rate(currentPruneRate): 
                                break
                        continue
                else:
                    continue
            
                # if filter has already been pruned, continue
                # could happen to due to dependencies
                if filterNum in self.channelsToPrune[layerName]:
                    continue 
                
                localRanking[layerName].pop(0)
                self.channelsToPrune[layerName].append(filterNum)
                currentPruneRate = self.inc_prune_rate(layerName, dependencies) 
            
            listIdx += 1
        
        print(f"Pruned blocks: {self.blocksToRemove}")
        return self.channelsToPrune
    #}}}
    
    def write_net(self, printFileLoc=True):
    #{{{
        if printFileLoc:
            print("Pruned model written to {}".format(self.filePath))
        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        
        if 'googlenet' in self.params.arch:
            self.writer = GoogLeNetWriter(self.netName, channelsPruned, self.depBlock, self.filePath,\
                    self.layerSizes)
        else:
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

