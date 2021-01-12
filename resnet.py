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
from functools import reduce

import torch
import torch.nn as nn

from base import BasicPruning

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
#}}}
