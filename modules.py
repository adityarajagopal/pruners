import torch
import torch.nn as nn

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

def residual_bottleneck(pruner, layerName, modName, mod, filterNum):
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

def se_residual_bottleneck(pruner, layerName, modName, mod, filterNum):
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
