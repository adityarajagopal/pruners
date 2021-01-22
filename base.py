import os
import sys
import csv
import time
import json
import math
import copy
import pickle
import functools
import importlib
import subprocess
from tqdm import tqdm
from collections import Counter
from abc import ABC, abstractmethod

# get current directory and append to path
# this allows everything inside pruners to access anything else 
# within pruners regardless of where pruners is
currDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currDir)

import dependencies as dependSrc
from model_writers import Writer, GoogLeNetWriter
from weight_transfer import WeightTransferUnit, ResNetWeightTransferUnit, GoogLeNetWeightTransferUnit

import numpy as np

import torch
import torch.nn as nn

import src.adapt.lego.utils as utils

class BasicPruning(ABC):
    def __init__(self, params, model, verbose=False):
    #{{{
        self.params = params
        self.model = model
        self.verbose = verbose
        self.importedPruNetModules = []

        self.minFiltersInLayer = 2

        self.metricValues = []
        self.blocksToRemove = []
        self.channelsToPrune = {}
        self.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]
        
        self.totalParams = 0
        self.layerSizes = {}
        
        # create model directory and file
        self.dirName = '{}/{}/{}'.format(params.pruner['model_path'], params.dataset,\
                params.pruner['subset_name'])
        self.filePath = os.path.join(self.dirName, self.fileName)
        
        ## create dir if it doesn't exist
        cmd = 'mkdir -p {}'.format(self.dirName)
        subprocess.check_call(cmd, shell=True)

        self.depBlock = dependSrc.DependencyBlock(model)
        self.get_layer_params()

        self.importPath = '{}.{}.{}'.format('.'.join(params.pruner['project_dir'].split('/')),\
                '.'.join(self.dirName.split('/')), self.fileName.split('.')[0])
    #}}} 
    
    def get_layer_params(self):
    #{{{
        for p in self.model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.totalParams += paramsInLayer
        
        self.notPruned = 0
        for n,m in self.model.named_modules(): 
            if self.is_conv_or_fc(m): 
                sizes = list(m._parameters['weight'].size())
                self.layerSizes["{}".format(n)] = sizes
            else: 
                if m._parameters:
                    self.notPruned += np.prod(m._parameters['weight'].size())
    #}}}

    def log_pruned_channels(self, rootFolder, params, totalPrunedPerc, channelsPruned): 
    #{{{
        if params.printOnly == True:
            return 
        
        # write pruned channels as json to log folder
        jsonName = os.path.join(rootFolder, 'pruned_channels.json')
        channelsPruned['prunePerc'] = totalPrunedPerc
        summary = {}
        summary[str(params.curr_epoch)] = channelsPruned
        with open(jsonName, 'w') as sumFile:
            json.dump(summary, sumFile)
        
        # copy pruned network description to log folder
        cmd = "cp {} {}/pruned_model.py".format(self.filePath, rootFolder)
        subprocess.check_call(cmd, shell=True)

        return summary
    #}}} 

    def get_random_init_model(self, finetunePath):    
    #{{{
        importPath = finetunePath.split('/')
        baseIdx = importPath.index('src')
        importPath = importPath[baseIdx:]
        importPath.append('pruned_model')
        self.importPath = '.'.join(importPath)
        prunedModel = self.import_pruned_model()
        optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        return prunedModel, optimiser
    #}}}

    def import_pruned_model(self):
    #{{{
        module = importlib.import_module(self.importPath) 
        if module not in self.importedPruNetModules:
            self.importedPruNetModules.append(module)
        else: 
            importlib.reload(module)
        pModel = module.__dict__[self.netName]
        prunedModel = pModel(num_classes=100)
        return prunedModel
    #}}}
    
    def prune_model(self, model, transferWeights=True, pruneNum=None):
    #{{{
        # pruning based on l1 norm of weights
        if self.params.pruner['mode'] == 'l1-norm':
            tqdm.write(f"Pruning filters: l1-norm - Pruning Level {self.params.pruner['pruning_perc']}")
            channelsPruned = self.structured_l1_weight_layer(model)\
                    if eval(self.params.pruner['prune_layers']) else self.structured_l1_weight(model)
            self.write_net()
            prunedModel = self.import_pruned_model()
            prunedModel = self.transfer_weights(model, prunedModel)
            pruneRate, prunedSize, origSize = self.prune_rate(prunedModel)
            print('Pruned Percentage = {:.2f}%, NewModelSize = {:.2f}MB, OrigModelSize = {:.2f}MB'.format(pruneRate, prunedSize, origSize))
            optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
            return channelsPruned, prunedModel, optimiser
        
        elif self.params.pruner['mode'] == 'random':
            tqdm.write("Pruning filters: random")
            channelsPruned = self.random_selection(model)
            self.write_net()
            prunedModel = self.import_pruned_model()
            prunedModel = self.transfer_weights(model, prunedModel)
            pruneRate, prunedSize, origSize = self.prune_rate(prunedModel)
            print('Pruned Percentage = {:.2f}%, NewModelSize = {:.2f}MB, OrigModelSize = {:.2f}MB'.format(pruneRate, prunedSize, origSize))
            optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
            return channelsPruned, prunedModel, optimiser
        
        elif self.params.pruner['mode'] == 'random_weighted': 
            tqdm.write("Pruning filters: random_weighted")
            if pruneNum is not None: 
                self.filePath = self.filePath.split('.')[0] + f"_{pruneNum}.py"
                self.importPath = '{}.{}.{}'.format('.'.join(self.params.pruner['project_dir'].split('/')), '.'.join(self.dirName.split('/')), f"{self.fileName.split('.')[0]}_{pruneNum}")
            channelsPruned = self.random_weighted_selection(model)
            self.write_net()
            prunedModel = self.import_pruned_model()
            pruneRate, prunedSize, origSize = self.prune_rate(prunedModel)
            print('Pruned Percentage = {:.2f}%, NewModelSize = {:.2f}MB, OrigModelSize = {:.2f}MB'.format(pruneRate, prunedSize, origSize))
            optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
            return channelsPruned, prunedModel, optimiser

     #}}}
        
    def non_zero_argmin(self, array): 
        minIdx = np.argmin(array[np.nonzero(array)]) 
        return (minIdx, array[minIdx])     
    
    def inc_prune_rate(self, layerName):
    #{{{
        lParam = str(layerName)
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1]*currLayerSize[2]*currLayerSize[3]
        # remove 1 output filter from current layer
        self.layerSizes[lParam][0] -= 1 
        
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
            
        self.currParams -= paramsPruned
        
        return (100. * (1. - (self.currParams / self.totalParams)))
    #}}}
    
    def prune_rate(self, pModel):
    #{{{
        prunedParams = 0
        for p in pModel.named_parameters():
            params = 1
            for dim in p[1].size():
                params *= dim 
            prunedParams += params
        
        return 100.*((self.totalParams - prunedParams) / self.totalParams), (prunedParams * 4) / 1e6, (self.totalParams * 4)/1e6
    #}}}        
    
    def inc_prune_rate(self, layerName):
    #{{{
        lParam = str(layerName)
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1] * currLayerSize[2] * currLayerSize[3]
        # remove 1 output filter from current layer
        self.layerSizes[lParam][0] -= 1 
        
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
            
        self.currParams -= paramsPruned
        
        return (100. * (1. - (self.currParams / self.totalParams)))
    #}}}
    
    def calculate_metric(self, param):
    #{{{
        #l1-norm
        metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
        metric /= (param.shape[1]*param.shape[2]*param.shape[3])
        return metric
    #}}}
    
    def rank_filters(self, model):
    #{{{
        localRanking = {} 
        globalRanking = []

        # create global ranking
        layers = []
        for p in model.named_parameters():
        #{{{
            layerName = '.'.join(p[0].split('.')[:-1])
            # if layerName in self.depBlock.linkedConvAndFc.keys() and layerName not in layers:
            if layerName in self.depBlock.linkedConvs.keys() and layerName not in layers:
                try:
                    netInst = type(self.model.module)
                    if layerName in self.depBlock.ignore[netInst]:
                        continue 
                except (AttributeError, KeyError): 
                    pass
                layers.append(layerName)
                
                pNp = p[1].data.cpu().numpy()
            
                # calculate metric
                metric = self.calculate_metric(pNp)

                globalRanking += [(layerName, i, x) for i,x in enumerate(metric)]
                localRanking[layerName] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])
        #}}}
        
        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        return localRanking, globalRanking
    #}}}
    
    def structured_l1_weight(self, model):
    #{{{
        localRanking, globalRanking = self.rank_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()
        
        pl = float(self.params.pruner['pruning_perc'])/100.
        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        
        if ('resnet' in self.params.arch or self.params.ofa is not None) and pl >= 0.85: 
            groupPruningLimits = [int(math.ceil(gs * 0.5)) for gs in numChannelsInDependencyGroup]
        else:
            ### This strategy ensures that at as many pruning levels as possible we maintain wider networks  
            ### At very high pruning levels (those in the if case above), we can't apply the same strategy
            ### as it limits the total amount of pruning possible (memory reduction) --> this is capped at 
            ### at most pruning 50% of the layer which allows desired memory reduction with some width maintenance
            ### This is enforced only for external dependencies, but internally within a block the internal layers
            ### can be pruned heavily
            if pl <= 0.5: 
                groupPruningLimits = [int(math.ceil(gs * (1.0 - pl))) for gs in numChannelsInDependencyGroup]
            else:
                groupPruningLimits = [int(math.ceil(gs * (pl))) for gs in numChannelsInDependencyGroup]
        
        groupPruningLimits = [self.minFiltersInLayer]*len(internalDeps) + groupPruningLimits
        
        dependencies = internalDeps + externalDeps
        self.remove_filters(localRanking, globalRanking, dependencies, groupPruningLimits)

        return self.channelsToPrune
    #}}}
    
    def select_random_weighted_filters(self, model):
    #{{{
        def generate_random_weighted_norms(param, layerMean):
            # k = layerMean * 2
            # theta = 0.5
            # dist = torch.distributions.gamma.Gamma(torch.Tensor([k]), torch.Tensor([theta]))
            dist = torch.distributions.uniform.Uniform(torch.Tensor([0]), torch.Tensor([1]))
            metric = dist.sample([param.shape[0]]).mul(layerMean)
            metric = metric.squeeze(1)
            return metric
        
        localRanking = {} 
        globalRanking = []

        numLayers = len(self.depBlock.linkedConvs.keys())
        
        choice = torch.rand(1)
        if choice < 0.3: 
            # increased late layer pruning
            layerMeans = [5*np.exp(-0.1*x) for x in range(numLayers)]
        elif choice >= 0.3 and choice < 0.6: 
            # increased early layer pruning
            layerMeans = [5/np.exp(-0.1*x) for x in range(numLayers)]
        else: 
            # uniform pruning
            layerMeans = [1 for x in range(numLayers)]

        # create global ranking
        layers = []
        layerCount = 0
        for p in model.named_parameters():
            layerName = '.'.join(p[0].split('.')[:-1])
            if layerName in self.depBlock.linkedConvs.keys() and layerName not in layers:
                try:
                    netInst = type(self.model.module)
                    if layerName in self.depBlock.ignore[netInst]:
                        continue 
                except (AttributeError, KeyError): 
                    pass
                layers.append(layerName)
                
                pNp = p[1].data.cpu().numpy()
            
                # calculate metric
                metric = generate_random_weighted_norms(pNp, layerMeans[layerCount])

                globalRanking += [(layerName, i, x) for i,x in enumerate(metric)]
                localRanking[layerName] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])
                layerCount += 1
        
        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        return localRanking, globalRanking
    #}}}
    
    def random_weighted_selection(self, model):
    #{{{
        localRanking, globalRanking = self.select_random_weighted_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()

        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        if float(self.params.pruner['pruning_perc']) >= 50.0:
            groupPruningLimits = [int(math.ceil(gs * (1.0 - float(self.params.pruner['pruning_perc'])/100.0))) for gs in numChannelsInDependencyGroup]
        else:
            groupPruningLimits = [int(math.ceil(gs * float(self.params.pruner['pruning_perc'])/100.0)) for gs in numChannelsInDependencyGroup]

        dependencies = internalDeps + externalDeps
        groupPruningLimits = [2]*len(internalDeps) + groupPruningLimits
        
        self.remove_filters(localRanking, globalRanking, dependencies, groupPruningLimits)

        return self.channelsToPrune
    #}}}
    
    def select_random_filters(self, model):
    #{{{
        def generate_random_norms(param):
            metric = torch.rand(param.shape[0])
            return metric
        
        localRanking = {} 
        globalRanking = []

        # create global ranking
        layers = []
        for p in model.named_parameters():
            layerName = '.'.join(p[0].split('.')[:-1])
            if layerName in self.depBlock.linkedConvs.keys() and layerName not in layers:
                netInst = type(self.model.module)
                try:
                    if layerName in self.depBlock.ignore[netInst]:
                        continue 
                except (AttributeError, KeyError): 
                    pass
                layers.append(layerName)
                
                pNp = p[1].data.cpu().numpy()
            
                # calculate metric
                metric = generate_random_norms(pNp)

                globalRanking += [(layerName, i, x) for i,x in enumerate(metric)]
                localRanking[layerName] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])
        
        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        return localRanking, globalRanking
    #}}}
    
    def random_selection(self, model):
    #{{{
        localRanking, globalRanking = self.select_random_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()

        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        if float(self.params.pruner['pruning_perc']) >= 50.0:
            groupPruningLimits = [int(math.ceil(gs * (1.0 - float(self.params.pruner['pruning_perc'])/100.0))) for gs in numChannelsInDependencyGroup]
        else:
            groupPruningLimits = [int(math.ceil(gs * float(self.params.pruner['pruning_perc'])/100.0)) for gs in numChannelsInDependencyGroup]

        dependencies = internalDeps + externalDeps
        groupPruningLimits = [2]*len(internalDeps) + groupPruningLimits
        
        self.remove_filters(localRanking, globalRanking, dependencies, groupPruningLimits)

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
                    
                    currentPruneRate = self.inc_prune_rate(layerName) 
            
            listIdx += 1
        
        print(f"Number of times min filter count hit = {count0}")
        return self.channelsToPrune
    #}}}
    
    def write_net(self):
    #{{{
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

        if 'googlenet' in self.params.arch:
            self.wtu = GoogLeNetWeightTransferUnit(self, pModStateDict, self.channelsToPrune, self.depBlock,\
                    self.layerSizes)
        else:
            self.wtu = WeightTransferUnit(self, pModStateDict, self.channelsToPrune, self.depBlock, self.layerSizes)
        
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
                    self.wtu.transfer_weights(type(m).__name__.lower(), n, m)
                except KeyError as e:
                    if self.verbose:
                        print(f"CRITICAL WARNING : layer found ({type(m)}) that is not handled in weight transfer. This could potentially break the network.")
        
        pModel.load_state_dict(pModStateDict)
        return pModel 
    #}}}
    
    # selects only convs and fc layers 
    def is_conv_or_fc(self, lModule):
    #{{{
        if isinstance(lModule, nn.Conv2d) or isinstance(lModule, nn.Linear):
            return True
        else:
            return False
    #}}}
