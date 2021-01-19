import sys
import copy
import time

import torch 
import itertools
import numpy as np
import torch.nn as nn

from sklearn.decomposition import PCA

class WeightTransferUnit(object): 
#{{{
    def __init__(self, app, prunedModel, channelsPruned, depBlk, layerSizes): 
    #{{{
        self.layerSizes = {k.replace('.','_'): v for k,v in layerSizes.items()}
        self.channelsPruned = {k.replace('.','_'): v for k,v in channelsPruned.items()}
        self.depBlk = depBlk
        self.pModel = prunedModel
        
        # resnet specific parameters
        self.enterResidual = False
        self.inMainBranch = False
        self.inDownsampleBranch = False
        self.prevBnStats = {'ofm_means': None, 'saved_means': None}
        
        self.device = f"cuda:{app.params.gpuList[0]}"

        # pre-fix checks if DataParallel wrapper is present around the model (module prefix) 
        self.prefix = '' if 'module' not in list(self.pModel.keys())[0] else 'module.'

        self.ipChannelsPruned = []

        baseMods = {'basic': nn_conv2d, 'relu': nn_relu, 'relu6': nn_relu6, 'maxpool2d':nn_maxpool2d,\
                    'avgpool2d':nn_avgpool2d, 'adaptiveavgpool2d':nn_adaptiveavgpool2d,\
                    'batchnorm2d': nn_batchnorm2d, 'linear': nn_linear, 'logsoftmax': nn_logsoftmax}
        # baseMods = {'basic': nn_conv2d_smart_update, 'relu': nn_relu, 'relu6': nn_relu6, 'maxpool2d':nn_maxpool2d,\
        #             'avgpool2d':nn_avgpool2d, 'adaptiveavgpool2d':nn_adaptiveavgpool2d,\
        #             'batchnorm2d': nn_batchnorm2d_smart_update, 'linear': nn_linear, 'logsoftmax': nn_logsoftmax}
        if hasattr(self, 'wtFuncs'):
            self.wtFuncs.update(baseMods)
        else:
            sef.wtFuncs = baseMods 
    #}}}

    @classmethod 
    def register_transfer_func(cls, blockName, func):
    #{{{
        if hasattr(cls, 'wtFuncs'):
            cls.wtFuncs[blockName] = func
        else: 
            setattr(cls, 'wtFuncs', {blockName:func})
    #}}}
        
    def transfer_weights(self, lType, modName, module): 
        self.wtFuncs[lType](self, modName, module)
#}}}

class GoogLeNetWeightTransferUnit(WeightTransferUnit): 
#{{{
    def __init__(self, app, prunedModel, channelsPruned, depBlk, layerSizes): 
        super().__init__(app, prunedModel, channelsPruned, depBlk, layerSizes)
        self.wtFuncs['linear'] = nn_linear_googlenet
#}}}

# torch.nn modules
def nn_conv2d_smart_update(wtu, modName, module, ipChannelsPruned=None, opChannelsPruned=None, dw=False): 
#{{{
    modName = modName.replace('.', '_')
    print(f"{modName}: enterRes-{wtu.enterResidual}; mainBr-{wtu.inMainBranch}; dsBr-{wtu.inDownsampleBranch}")
    bnKey = 'ofm_means'
    if wtu.enterResidual: 
        wtu.enterResidual = False
        wtu.prevBnStats['saved_means'] = copy.deepcopy(wtu.prevBnStats['ofm_means'])
    if wtu.inDownsampleBranch: 
        bnKey = 'saved_means'
    
    dw = dw or (module.in_channels == module.groups)
    allIpChannels = list(range(module.in_channels))
    allOpChannels = list(range(module.out_channels))
    ipPruned = wtu.ipChannelsPruned if ipChannelsPruned is None else ipChannelsPruned
    opPruned = wtu.channelsPruned[modName] if opChannelsPruned is None else opChannelsPruned
    ipChannels = list(set(allIpChannels) - set(ipPruned)) if not dw else [0]
    opChannels = list(set(allOpChannels) - set(opPruned))
    wtu.ipChannelsPruned = opPruned 
    
    pWeight = f'{wtu.prefix}{modName}.weight'
    pBias = f'{wtu.prefix}{modName}.bias'
    
    if wtu.prevBnStats[bnKey] is not None:
        wtu.pModel[pWeight] = module._parameters['weight'][opChannels,:][:,ipChannels]
        print(f"opChannels = {len(opChannels)}, ipChannelsPruned = {len(ipPruned)}")
        
        # for o,_opC in enumerate(opChannels):
        #     for i,_ipC in enumerate(ipChannels): 
        #         ui = wtu.prevBnStats[bnKey][_ipC]
        #         ai = 1. / (ui * len(ipChannels))
        #         _sum = None
        #         for _prunedIpC in ipPruned: 
        #             ap = module._parameters['weight'][_opC][_prunedIpC].mul(\
        #                     wtu.prevBnStats[bnKey][_prunedIpC])
        #             if _sum is None:
        #                 _sum = ap
        #             else:
        #                 _sum = _sum.add(ap)
        #         if _sum is not None:
        #             wtu.pModel[pWeight][o][i].add_(_sum.mul(ai))   
        
        # for i,_ipC in enumerate(ipChannels): 
        #     ui = wtu.prevBnStats[bnKey][_ipC]
        #     ai = 1. / (ui * len(ipChannels))
        #     _sum = None
        #     for _prunedIpC in ipPruned: 
        #         ap = module._parameters['weight'][opChannels,:][:,_prunedIpC].mul(\
        #                 wtu.prevBnStats[bnKey][_prunedIpC])
        #         if _sum is None:
        #             _sum = ap
        #         else:
        #             _sum = _sum.add(ap)
        #     if _sum is not None:
        #         wtu.pModel[pWeight][:,i].add_(_sum.mul(ai))   
        
        if len(ipPruned) != 0:
            linComb = torch.nn.Conv2d(len(ipPruned), 1, kernel_size=(1,1), bias=False)
            linComb.weight.detach_()
            for i,_ipC in enumerate(ipChannels): 
                ui = wtu.prevBnStats[bnKey][_ipC]
                ai = 1. / (ui * len(ipChannels))
                _sum = None
                with torch.no_grad():
                    linComb.weight.copy_(\
                            wtu.prevBnStats[bnKey][ipPruned].unsqueeze(0).unsqueeze(2).unsqueeze(3)\
                            .type_as(linComb.weight))
                linComb.to(wtu.device)
                fil = module._parameters['weight'][opChannels,:][:,ipPruned].to(wtu.device)
                _sum = linComb(fil).squeeze(1).cpu()
                wtu.pModel[pWeight][:,i].add_(_sum.mul(ai))   
        
    if module._parameters['bias'] is not None:
        wtu.pModel[pBias] = module._parameters['bias'][opChannels]

    wtu.prevLayer = modName
#}}}

def nn_batchnorm2d_smart_update(wtu, modName, module): 
#{{{
    modName = modName.replace('.', '_')
    print(f"{modName}: enterRes-{wtu.enterResidual}; mainBr-{wtu.inMainBranch}; dsBr-{wtu.inDownsampleBranch}")
    if wtu.inDownsampleBranch:
        wtu.prevBnStats['ofm_means'] += module.weight.detach()
    else:
        wtu.prevBnStats['ofm_means'] = module.weight.detach()
    
    allFeatures = list(range(module.num_features))
    numFeaturesKept = list(set(allFeatures) - set(wtu.ipChannelsPruned))
    
    key = f'{wtu.prefix}{modName}'
    wtu.pModel['{}.weight'.format(key)] = module._parameters['weight'][numFeaturesKept]
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias'][numFeaturesKept]
    wtu.pModel['{}.running_mean'.format(key)] = module._buffers['running_mean'][numFeaturesKept]
    wtu.pModel['{}.running_var'.format(key)] = module._buffers['running_var'][numFeaturesKept]
    wtu.pModel['{}.num_batches_tracked'.format(key)] = module._buffers['num_batches_tracked']
#}}}

def nn_conv2d(wtu, modName, module, ipChannelsPruned=None, opChannelsPruned=None, dw=False): 
#{{{
    modName = modName.replace('.', '_')
    
    dw = dw or (module.in_channels == module.groups)
    allIpChannels = list(range(module.in_channels))
    allOpChannels = list(range(module.out_channels))
    ipPruned = wtu.ipChannelsPruned if ipChannelsPruned is None else ipChannelsPruned
    opPruned = wtu.channelsPruned[modName] if opChannelsPruned is None else opChannelsPruned
    ipChannels = list(set(allIpChannels) - set(ipPruned)) if not dw else [0]
    opChannels = list(set(allOpChannels) - set(opPruned))
    wtu.ipChannelsPruned = opPruned 
    
    pWeight = f'{wtu.prefix}{modName}.weight'
    pBias = f'{wtu.prefix}{modName}.bias'
    wtu.pModel[pWeight] = module._parameters['weight'][opChannels,:][:,ipChannels]
    if module._parameters['bias'] is not None:
        wtu.pModel[pBias] = module._parameters['bias'][opChannels]

    wtu.prevLayer = modName
#}}}

def nn_batchnorm2d(wtu, modName, module): 
#{{{
    modName = modName.replace('.', '_')
    
    allFeatures = list(range(module.num_features))
    numFeaturesKept = list(set(allFeatures) - set(wtu.ipChannelsPruned))
    
    key = f'{wtu.prefix}{modName}'
    wtu.pModel['{}.weight'.format(key)] = module._parameters['weight'][numFeaturesKept]
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias'][numFeaturesKept]
    wtu.pModel['{}.running_mean'.format(key)] = module._buffers['running_mean'][numFeaturesKept]
    wtu.pModel['{}.running_var'.format(key)] = module._buffers['running_var'][numFeaturesKept]
    wtu.pModel['{}.num_batches_tracked'.format(key)] = module._buffers['num_batches_tracked']
#}}}

def nn_linear(wtu, modName, module): 
#{{{
    modName = modName.replace('.', '_')
    
    inFeatures = wtu.layerSizes[modName][1]
    prevOutChannels = wtu.layerSizes[wtu.prevLayer][0]
    spatialSize = int(inFeatures / prevOutChannels)

    allIpChannels = list(range(int(module.in_features/spatialSize)))
    ipChannelsKept = list(set(allIpChannels) - set(wtu.ipChannelsPruned))
    
    key = f'{wtu.prefix}{modName}'
    for i,channel in enumerate(ipChannelsKept):
        sOrig = spatialSize * channel
        eOrig = sOrig + spatialSize
        sPrune = spatialSize * i
        ePrune = sPrune + spatialSize
        wtu.pModel['{}.weight'.format(key)][:,sPrune:ePrune] = module._parameters['weight'][:,sOrig:eOrig]
    
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias']

    wtu.ipChannelsPruned = []
    wtu.prevLayer = modName
#}}}

def nn_linear_googlenet(wtu, modName, module): 
#{{{
    modName = modName.replace('.', '_')
    
    inFeatures = wtu.layerSizes[modName][1]
    _depBlk = {k.replace('.','_'): [(x[0].replace('.','_'), x[1]) for x in v] for k,v in\
            wtu.depBlk.linkedConvAndFc.items()}
    prevLayers = [k for k,v in _depBlk.items() if v[0][0] == modName]
    prevOutChannels = sum([wtu.layerSizes[x][0] for x in prevLayers]) 

    spatialSize = int(inFeatures / prevOutChannels)

    allIpChannels = list(range(int(module.in_features/spatialSize)))
    ipChannelsKept = list(set(allIpChannels) - set(wtu.ipChannelsPruned))
    
    key = f'{wtu.prefix}{modName}'
    for i,channel in enumerate(ipChannelsKept):
        sOrig = spatialSize * channel
        eOrig = sOrig + spatialSize
        sPrune = spatialSize * i
        ePrune = sPrune + spatialSize
        wtu.pModel['{}.weight'.format(key)][:,sPrune:ePrune] = module._parameters['weight'][:,sOrig:eOrig]
    
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias']

    wtu.ipChannelsPruned = []
    wtu.prevLayer = modName
#}}}

def nn_relu(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_relu6(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_maxpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_avgpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_adaptiveavgpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def ofa_adaptiveavgpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_logsoftmax(wtu, modName, module): 
#{{{
    pass
#}}}

# custom modules
def residual_backbone(wtu, modName, module, main_branch, residual_branch, aggregation_op):
#{{{
    inputToBlock = wtu.ipChannelsPruned
    idx = wtu.depBlk.instances.index(type(module))

    wtu.enterResidual = True
    
    # main path through residual
    for n,m in module.named_modules(): 
        wtu.inMainBranch = True
        fullName = "{}.{}".format(modName, n)
        if not any(x in n for x in wtu.depBlk.dsLayers[idx]):
            main_branch(n, m, fullName, wtu)
    wtu.inMainBranch = False
    
    outputOfBlock = wtu.ipChannelsPruned

    # downsampling path if exists
    if residual_branch is not None:
        wtu.inDownsampleBranch = True 
        for n,m in module.named_modules(): 
            fullName = "{}.{}".format(modName, n)
            if any(x in n for x in wtu.depBlk.dsLayers[idx]):
                residual_branch(n, m, fullName, wtu, inputToBlock, outputOfBlock)
        wtu.inDownsampleBranch = False
        
        if aggregation_op is not None:
            aggregation_op(wtu, opNode, opNode1)
#}}}

def split_and_aggregate_backbone(wtu, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs), 'For each branch a processing function must be provided - branches = {}, procFuns = {}'.format(len(branchConvs), len(branchProcs))

    inputToBlock = wtu.ipChannelsPruned

    branchOpChannels = []
    opNodes = None
            
    for idx in range(len(branchStarts)):
        wtu.ipChannelsPruned = inputToBlock

        inBranch = False
        for n,m in parentModule._modules.items(): 
            if n in branchStarts and not inBranch:
                inBranch = True
                branchStarts.pop(0)
            elif n in branchStarts and inBranch: 
                break
            
            if inBranch:
                fullName = "{}.{}".format(parentModName, n)
                branchProcs[idx](wtu, fullName, m)
        
        branchOpChannels.append(wtu.ipChannelsPruned)

    if aggregation_op is not None:
        aggregation_op(wtu, opNodes, branchOpChannels)  
#}}}

def residual(wtu, modName, module):
#{{{
    def main_branch(n, m, fullName, wtu): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m)
            # start = time.time()
            # nn_conv2d_smart_update(wtu, fullName, m)
            # print(f"Update took {time.time() - start}")

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, m)
            # nn_batchnorm2d_smart_update(wtu, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, wtu, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m, ipToBlock, opOfBlock)
            # start = time.time()
            # nn_conv2d_smart_update(wtu, fullName, m, ipToBlock, opOfBlock)
            # print(f"Update took {time.time() - start}")
            
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
            # nn_batchnorm2d_smart_update(wtu, fullName, m)
    #}}}

    residual_backbone(wtu, modName, module, main_branch, residual_branch, None)
#}}}

def ofa_residual(wtu, modName, module):
#{{{
    def main_branch(n, m, fullName, wtu): 
    #{{{
        fullName = fullName.replace('.','_')
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m)

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, wtu, ipToBlock, opOfBlock): 
    #{{{
        fullName = fullName.replace('.', '_')
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m, ipToBlock, opOfBlock)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
    #}}}

    residual_backbone(wtu, modName, module, main_branch, residual_branch, None)
#}}}

def mb_conv(wtu, modName, module):
#{{{
    def main_branch(n, m, fullName, wtu): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = wtu.depBlk.instances.index(type(module))
            wtu.convIdx = wtu.depBlk.convs[idx].index(n)
            nn_conv2d(wtu, fullName, m, None, None, dw=(wtu.convIdx==1))

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, wtu, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m, ipToBlock, opOfBlock, dw=False)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    idx = wtu.depBlk.instances.index(type(module))
    midConv = wtu.depBlk.convs[idx][1] 
    # stride = module._modules[midConv].stride[0]
    stride = dict(module.named_modules())[midConv].stride[0]
    if stride == 2: 
        residual_backbone(wtu, modName, module, main_branch, None, None)
    else:
        residual_backbone(wtu, modName, module, main_branch, residual_branch, None)
#}}}

def fire(wtu, modName, module): 
#{{{
    wtu.totalOutChannels = []
    def basic(wtu, fullName, module): 
    #{{{
        if isinstance(module, nn.Conv2d): 
            wtu.totalOutChannels.append(module.out_channels)
            nn_conv2d(wtu, fullName, module)

        if isinstance(module, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, module)
    #}}}

    def aggregation_op(wtu, opNodes, branchOpChannels): 
    #{{{
        offsets = [0] + list(np.cumsum(wtu.totalOutChannels))[:-1]
        prunedChannels = [list(offsets[i] + np.array(x)) for i,x in enumerate(branchOpChannels)]    
        prunedChannels = list(itertools.chain.from_iterable(prunedChannels))
        wtu.ipChannelsPruned = prunedChannels
    #}}}
    
    idx = wtu.depBlk.instances.index(type(module))
    convs = wtu.depBlk.convs[idx]

    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if isinstance(m, nn.Conv2d): 
            if n == convs[0]:
                nn_conv2d(wtu, fullName, m)
        
        if isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
        
        if isinstance(m, nn.ReLU): 
            split_and_aggregate_backbone(wtu, modName, module, convs[1:], [basic,basic], aggregation_op)
            break
#}}}

def split_and_aggregate_backbone_new(wtu, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs),\
            'For each branch a processing function must be provided - branches = {}, procFuns = {}'\
            .format(len(branchConvs), len(branchProcs))

    inputToBlock = wtu.ipChannelsPruned

    branchOpChannels = []
    opNodes = None
            
    for idx, branchName in enumerate(branchStarts):
        wtu.ipChannelsPruned = inputToBlock
        fullName = "{}.{}".format(parentModName, branchName)
        branchProcs[idx](wtu, fullName, dict(parentModule.named_modules())[branchName])
        branchOpChannels.append(wtu.ipChannelsPruned)

    if aggregation_op is not None:
        aggregation_op(wtu, opNodes, branchOpChannels)  
#}}}

def inception(wtu, modName, module): 
#{{{
    wtu.totalOutChannels = []
    def basic(wtu, fullName, module): 
    #{{{
        for n,m in module.named_modules():
            name = f"{fullName}.{n}"	 	
            if isinstance(m, nn.Conv2d): 
                # check if next layer is still within the same branch
                # only append to out channels if it is an exit layer of the branch
                # TODO: Incorporate this as a dependency block function
                nextLayers = wtu.depBlk.linkedConvAndFc[name]
                if not all(f'{fullName}.' in x[0] for x in nextLayers):
                    wtu.totalOutChannels.append(m.out_channels)
                nn_conv2d(wtu, name, m)

            if isinstance(m, nn.BatchNorm2d): 
                nn_batchnorm2d(wtu, name, m)
    #}}}

    def aggregation_op(wtu, opNodes, branchOpChannels): 
    #{{{
        offsets = [0] + list(np.cumsum(wtu.totalOutChannels))[:-1]
        prunedChannels = [list(offsets[i] + np.array(x)) for i,x in enumerate(branchOpChannels)]    
        prunedChannels = list(itertools.chain.from_iterable(prunedChannels))
        wtu.ipChannelsPruned = prunedChannels
    #}}}
    
    idx = wtu.depBlk.instances.index(type(module))
    convs = wtu.depBlk.convs[idx]
    split_and_aggregate_backbone_new(wtu, modName, module, convs, [basic,basic,basic,basic], aggregation_op)
#}}}
