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
from src.adapt.lego.pruners.base import BasicPruning

class ResNet20Pruning(BasicPruning):
#{{{
    def __init__(self, params, model):  
        self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruningPerc))
        self.netName = 'ResNet{}'.format(int(params.depth))
        super().__init__(params, model)

    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
        fprint('class {}(nn.Module):'.format(self.netName))
        fprint('\tdef __init__(self, num_classes=10):')
        fprint('\t\tsuper().__init__()')
        fprint('')

        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        start = True
        currentIpChannels = 3

        linesToWrite = {}
        for n,m in self.model.named_modules():
        #{{{
            if not m._modules:
                if 'downsample' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    elif isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels

                    elif isinstance(m, nn.ReLU):
                        continue
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                blockInChannels[n] = (m._modules['conv1'].in_channels, m._modules['conv2'].out_channels, m._modules['conv1'].stride)
        
        self.orderedKeys = list(linesToWrite.keys())
        for k,v in blockInChannels.items():
            if v[0] == v[1]:
                newKey = k + '.downsample'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Sequential()
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = k + '.downsample.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Conv2d(v[0], v[1], kernel_size=1, stride=v[2], padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.downsample.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.downsample.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

        [fprint(linesToWrite[k]) for k in self.orderedKeys]
        
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layer' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                if 'downsample.0' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif 'downsample' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(x))'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+1
                else:
                    fprint('\t\tx = F.relu(out)')
            
            elif 'fc' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
            
            elif 'avgpool' in self.orderedKeys[i]:
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

        fprint('\t\treturn x')
        fprint('')
        # fprint('def resnet_{}(**kwargs):'.format(self.params.pruningPerc))
        # fprint('\treturn ResNet_{}(**kwargs)'.format(self.params.pruningPerc))
        fprint('def resnet(**kwargs):')
        fprint('\treturn ResNet(**kwargs)')
        #}}}                  

        self.modelDesc.close()
    #}}}
    
    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 

        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []
        for k in self.orderedKeys:
            if 'conv' in k:
            #{{{
                layer = k
                param = k + '.weight'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmp[:,ipChannelsKept] 
                
                ipChannelsToPrune = opChannelsToPrune
            #}}}
            
            elif 'bn' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
            #}}}
            
            elif 'fc' in k:
            #{{{
                layer = k
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                
                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
            #}}}

        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}

class ResNet20PruningConcat(BasicPruning):
#{{{
    def __init__(self, params, model):  
        self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruningPerc))
        self.netName = 'ResNet{}'.format(int(params.depth))
        super().__init__(params, model)

    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')
        # self.modelDesc = sys.stdout 

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
        fprint('class {}(nn.Module):'.format(self.netName))
        fprint('\tdef __init__(self, num_classes=10):')
        fprint('\t\tsuper().__init__()')
        fprint('')

        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        start = True
        currentIpChannels = 3
        
        linesToWrite = {}
        for n,m in self.model.named_modules():
        #{{{
            if not m._modules:
                if 'downsample' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    if isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels

                    elif isinstance(m, nn.ReLU):
                        continue
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                blockInChannels[n] = [m._modules['conv1'].in_channels, m._modules['conv2'].out_channels, m._modules['conv1'].stride]
        
        self.orderedKeys = list(linesToWrite.keys())
        updatedIpChannels = {k:v[0] for k,v in blockInChannels.items()}
        for i,(k,v) in enumerate(blockInChannels.items()):

            ic = updatedIpChannels[k]
            
            if v[2][0] == 2:
                newKey = k + '.downsample.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Conv2d(ic, v[1], kernel_size=1, stride=v[2], padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.downsample.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.downsample.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = 'ignore:res' if ic == v[1] else 'ignore:pad_out' if ic > v[1] else 'ignore:pad_x'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2') + 1, newKey)
                
                if newKey == 'ignore:pad_out':
                    toEdit = self.orderedKeys[self.orderedKeys.index(k + '.bn2') + 2]
                    lineToEdit = linesToWrite[toEdit]
                    modName, module = lineToEdit.split('=',1)
                    module = eval(module)
                    module.in_channels = v[0]
                    linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

                    layerName = '.'.join(toEdit.split('.')[:-1])
                    updatedIpChannels[layerName] = v[0]
        
        # if last layer has a concatenated output, then fc layer that follows this needs to also have a different input size
        if newKey == 'ignore:pad_out':
            i = self.orderedKeys.index(k + '.bn2')
            while('fc' not in self.orderedKeys[i]):
                i += 1
            toEdit = self.orderedKeys[i]
            lineToEdit = linesToWrite[toEdit]
            modName, module = lineToEdit.split('=',1)
            module = eval(module)
            module.in_features = ic
            linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

        [fprint(linesToWrite[k]) for k in self.orderedKeys if 'ignore' not in k]

        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layer' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                
                if 'downsample.0' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif self.orderedKeys[i] == 'ignore:res':
                    i += 1
                elif self.orderedKeys[i] == 'ignore:pad_x':
                    fprint("\t\ttmp = torch.zeros(out.shape[0], out.shape[1] - x.shape[1], out.shape[2], out.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tx = torch.cat([x, tmp], dim=1)")
                    i += 1 
                elif self.orderedKeys[i] == 'ignore:pad_out':
                    fprint("\t\ttmp = torch.zeros(x.shape[0], x.shape[1] - out.shape[1], x.shape[2], x.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tout = torch.cat([out, tmp], dim=1)")
                    i += 1 
                
                fprint('\t\tx = F.relu(out + x)')
            
            elif 'fc' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
            
            elif 'avgpool' in self.orderedKeys[i]:
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

        fprint('\t\treturn x')
        fprint('')
        # fprint('def resnet_{}(**kwargs):'.format(self.params.pruningPerc))
        # fprint('\treturn ResNet_{}(**kwargs)'.format(self.params.pruningPerc))
        fprint('def resnet(**kwargs):')
        fprint('\treturn ResNet(**kwargs)')
        #}}}                  

        self.modelDesc.close()
    #}}}
    
    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 

        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []

        for k in self.orderedKeys:
            if 'conv' in k:
            #{{{
                layer = k
                param = k + '.weight'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam][:,:len(ipChannelsKept)] = tmp[:,ipChannelsKept] 

                ipChannelsToPrune = opChannelsToPrune
            #}}}
            
            elif 'bn' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
            #}}}
            
            elif 'fc' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW][:,:len(opChannelsKept)] = parentModel[paramW][:,opChannelsKept]
            #}}}

        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}

class ResNet20PruningDependency(BasicPruning):
#{{{
    def __init__(self, params, model):  
    #{{{
        if params.ofa is not None and params.ofa['task'] == 'prune_and_retrain': 
            self.fileName = 'ofa_{}_{}.py'.format(params.ofa['network'], params.pruner['pruning_perc'])
            self.netName = f"OFA{params.ofa['network'].capitalize()}"
        else:
            self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruner['pruning_perc']))
            self.netName = 'ResNet{}'.format(int(params.depth))
        
        super().__init__(params, model)
    #}}}
    
    def inc_prune_rate(self, layerName, dependencies):
    #{{{
        lParam = str(layerName)
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1] * currLayerSize[2] * currLayerSize[3]
        # remove 1 output filter from current layer
        self.layerSizes[lParam][0] -= 1 
        
        # check if it is at the head of a dependency group, i.e. it has a downsampling layer
        if any(x.index(layerName) == 0 for x in dependencies if layerName in x):
            blockName = '.'.join(layerName.split('.')[:-1])
            module = [m for n,m in self.model.named_modules() if n == blockName][0]
            instances = [isinstance(module, x) for x in self.depBlock.instances]
            if True in instances: 
                instanceIdx = instance.index(True)
                dsInstName = self.depBlock.dsLayers[instanceIdx][0]
                dsLayer = [x for x,p in module.named_modules() if dsInstName in x and\
                        isinstance(p, torch.nn.Conv2d)][0]
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
            
        self.currParams -= paramsPruned
        
        return (100. * (1. - (self.currParams / self.totalParams)))
    #}}}
    
    def inc_prune_rate_block(self, blockName, localRanking, dependencies):
    #{{{
        ### update prune rate
        for l,nl in self.depBlock.linkedConvAndFc.items(): 
            if blockName in l:
                channelsLeftToPrune = len(localRanking[l])
                del localRanking[l]
                del self.channelsToPrune[l]
                currentPruneRate = [self.inc_prune_rate(l, dependencies) for i in range(channelsLeftToPrune)][-1]

        ### update layerSizes        
        module = [m for n,m in self.model.named_modules() if n == blockName][0]
        instanceIdx = [isinstance(module, x) for x in self.depBlock.instances].index(True)
        internalLayerNames = self.depBlock.convs[instanceIdx]
        dsLayerName = self.depBlock.dsLayers[instanceIdx][0]
            
        prevBlock = [self.depBlock.linkedModules[i-1][1] for i,x in enumerate(self.depBlock.linkedModules)\
                if x[1] == blockName][0]
        nextBlock = [self.depBlock.linkedModules[i+1][1] for i,x in enumerate(self.depBlock.linkedModules)\
                if x[1] == blockName][0]
        currIpChannels = self.layerSizes[f"{blockName}.{internalLayerNames[0]}"][1]
        currOpChannels = self.layerSizes[f"{nextBlock}.{internalLayerNames[2]}"][0]
        
        prevLayer = f"{prevBlock}.{internalLayerNames[2]}"
        currLayer = f"{blockName}.{internalLayerNames[2]}"
        nextLayer = f"{nextBlock}.{internalLayerNames[0]}"

        for nLayer,_ in self.depBlock.linkedConvAndFc[currLayer]:
            self.layerSizes[nLayer][1] = currIpChannels
        
        newLS = copy.deepcopy(self.layerSizes)
        for k,v in self.layerSizes.items(): 
            if blockName in k: 
                del newLS[k]
        self.layerSizes = newLS

        ### update linked modules
        newLM = [x for x in self.depBlock.linkedModules if x[1] != blockName]
        self.depBlock.linkedModules = newLM
        
        ### update linkedConvAndFc and linkedConvs
        newLCF = {k:v for k,v in self.depBlock.linkedConvAndFc.items() if blockName not in k}
        newLCF[prevLayer] = self.depBlock.linkedConvAndFc[currLayer]
        self.depBlock.linkedConvAndFc = newLCF
        self.depBlock.linkedConvs = self.depBlock.linkedConvAndFc
        
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
        listIdx = 0
        currentPruneRate = 0
        self.currParams = self.totalParams
        while (currentPruneRate < float(self.params.pruner['pruning_perc'])) and (listIdx < len(globalRanking)):
            layerName, filterNum, _ = globalRanking[listIdx]

            depLayers = []
            pruningLimit = self.minFiltersInLayer
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    pruningLimit = groupPruningLimits[i][group.index(layerName)]
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
                for i,layerName in enumerate(depLayers):
                    # case where you want to skip layers
                    # if layers are dependent, skipping one means skipping all the dependent layers
                    # i == 0 prevents us from removing head block from dependency group as it has stride 2 and
                    # downsamples the activations --> necessary
                    if (layerName in localRanking.keys()):
                        if (len(localRanking[layerName]) <= pruningLimit):
                            blockName = '.'.join(layerName.split('.')[:-1])
                            if i != 0 and blockName not in self.blocksToRemove:
                                self.blocksToRemove.append(blockName)
                                currentPruneRate = self.inc_prune_rate_block(blockName, localRanking, dependencies)
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
        
        print(f"Blocks to remove: {self.blocksToRemove}")
        return self.channelsToPrune
    #}}}
#}}}
