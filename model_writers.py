import sys
import copy

import torch.nn as nn
    
class Writer(object): 
#{{{
    def __init__(self, netName, channelsPruned, depBlk, modelFile, layerSizes, forwardVar='x'): 
    #{{{
        self.currIpChannels = 3
        self.channelsPruned = channelsPruned
        self.depBlk = depBlk
        self.forVar = forwardVar
        self.netName = netName
        self.toWrite = {'modules':[], 'forward':[]}
        self.modelDesc = open(modelFile, 'w+')
        self.layerSizes = layerSizes

        self.useCount = False 
        self.returnStatement = "return {}.squeeze(2).squeeze(2)" if 'squeezenet' == self.netName.lower() else "return {}"
        
        baseWriters = {'basic': nn_conv2d, 'relu': nn_relu, 'relu6': nn_relu6, 'maxpool2d':nn_maxpool2d,\
                       'avgpool2d':nn_avgpool2d, 'adaptiveavgpool2d':nn_adaptiveavgpool2d,\
                       'batchnorm2d': nn_batchnorm2d, 'linear': nn_linear, 'logsoftmax': nn_logsoftmax}
        if hasattr(self, 'writers'):
            self.writers.update(baseWriters)
        else:
            sef.writers = baseWriters 
    #}}}

    @classmethod 
    def register_writer(cls, blockName, func):
    #{{{
        if hasattr(cls, 'writers'):
            cls.writers[blockName] = func
        else: 
            setattr(cls, 'writers', {blockName:func})
    #}}}
        
    def fprint(self, text):
        print(text.replace('\t','    '), file=self.modelDesc)

    def write_preamble(self):
    #{{{
        self.fprint('import torch')
        self.fprint('import torch.nn as nn')
        self.fprint('import torch.nn.functional as F')
    
        self.fprint('')
        self.fprint('class {}(nn.Module):'.format(self.netName))
        self.fprint('\tdef __init__(self, num_classes=10):')
        self.fprint('\t\tsuper().__init__()')
        self.fprint('')
    #}}}

    def write_postamble(self):
    #{{{
        ret = self.returnStatement.format(self.forVar)
        self.fprint('\t\t{}'.format(ret))
        self.fprint('')
        self.fprint('def {}(**kwargs):'.format(self.netName.lower()))
        self.fprint('\treturn {}(**kwargs)'.format(self.netName))
    #}}}

    def write_network(self, stdout=False): 
    #{{{
        if stdout:
            self.modelDesc = sys.stdout 

        self.write_preamble()
        
        [self.fprint(x) for x in self.toWrite['modules']]
        
        self.fprint('')
        self.fprint('\tdef forward(self, {}):'.format(self.forVar))
        
        [self.fprint(x) for x in self.toWrite['forward']]

        self.write_postamble()

        self.modelDesc.close()
    #}}}
    
    def write_module(self, className, modName, module): 
    #{{{
        self.writers[className](self, modName, module)
    #}}}

    def write_module_desc(self, modName, module):
    #{{{
        layerName = modName.replace('.','_')
        mod = '\t\tself.{} = nn.{}'.format(layerName, str(module))
        self.toWrite['modules'].append(mod)
    #}}}

    def write_module_forward(self, modName):
    #{{{
        layerName = modName.replace('.','_')
        forward = '\t\t{} = self.{}({})'.format(self.forVar, layerName, self.forVar) 
        self.toWrite['forward'].append(forward)
    #}}}
#}}}

class GoogLeNetWriter(Writer): 
#{{{
    def _transform_input_fn(self): 
    #{{{
        self.fprint(f"\tdef transform_input(self, {self.forVar}):")
        self.fprint(f"\t\tx_ch0 = torch.unsqueeze({self.forVar}[:,0],1) * (0.229/0.5) + (0.485-0.5) / 0.5")
        self.fprint(f"\t\tx_ch1 = torch.unsqueeze({self.forVar}[:,1],1) * (0.224/0.5) + (0.456-0.5) / 0.5")
        self.fprint(f"\t\tx_ch2 = torch.unsqueeze({self.forVar}[:,2],1) * (0.225/0.5) + (0.406-0.5) / 0.5")
        self.fprint(f"\t\t{self.forVar} = torch.cat((x_ch0, x_ch1, x_ch2), 1)")

        self.fprint(f"\t\treturn {self.forVar}")
    #}}}

    def write_network(self, stdout=False): 
    #{{{
        if stdout:
            self.modelDesc = sys.stdout 

        self.write_preamble()
        
        [self.fprint(x) for x in self.toWrite['modules']]
        
        self.fprint('')
        self._transform_input_fn()

        self.fprint('')
        self.fprint('\tdef forward(self, {}):'.format(self.forVar))
        
        self.toWrite['forward'] = [f"\t\t{self.forVar} = self.transform_input({self.forVar})"]\
                + self.toWrite['forward']
        [self.fprint(x) for x in self.toWrite['forward']]

        self.write_postamble()

        self.modelDesc.close()
    #}}}
#}}}

class OFAResNetWriter(Writer): 
#{{{
    def write_module_desc(self, modName, module):
    #{{{
        mod = '\t\tself.{} = nn.{}'.format(modName.replace('.', '_'), str(module))
        self.toWrite['modules'].append(mod)
    #}}}

    def write_module_forward(self, modName):
    #{{{
        forward = '\t\t{} = self.{}({})'.format(self.forVar, modName.replace('.','_'), self.forVar) 
        self.toWrite['forward'].append(forward)
    #}}}
#}}}

# torch.nn modules
def nn_conv2d(writer, modName, module, dw=False): 
#{{{
    # print(dw, module.groups, writer.currIpChannels, writer.layerSizes[modName][1])
    dw = dw or (module.in_channels == module.groups)
    writer.currIpChannels = writer.layerSizes[modName][1] if not dw else writer.currIpChannels
    module.in_channels = writer.currIpChannels 
    if dw:
        module.groups = writer.currIpChannels
    # module.out_channels -= writer.channelsPruned[modName]
    module.out_channels = writer.layerSizes[modName][0] 
    writer.currIpChannels = module.out_channels
    
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_relu(writer, modName, module): 
#{{{
    writer.toWrite['forward'].append('\t\t{} = F.relu({}, inplace=True)'.format(writer.forVar, writer.forVar)) 
#}}}

def nn_relu6(writer, modName, module): 
#{{{
    writer.toWrite['forward'].append('\t\t{} = F.relu6({}, inplace=True)'.format(writer.forVar, writer.forVar)) 
#}}}

def nn_maxpool2d(writer, modName, module): 
#{{{
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_avgpool2d(writer, modName, module): 
#{{{
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_adaptiveavgpool2d(writer, modName, module): 
#{{{
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def ofa_adaptiveavgpool2d(writer, modName, module): 
#{{{
    module = nn.AdaptiveAvgPool2d((1,1))
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_batchnorm2d(writer, modName, module): 
#{{{
    module.num_features = writer.currIpChannels
    
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_linear(writer, modName, module): 
#{{{
    if writer.useCount:
        # needed in the case of googlenet where concatenation means
        # layerSizes is incorrect for FC
        module.in_features = writer.currIpChannels
    else:
        module.in_features = writer.layerSizes[modName][1] 
    
    writer.write_module_desc(modName, module)
    
    writer.toWrite['forward'].append('\t\t{} = {}.view({}.size(0), -1)'.format(writer.forVar, writer.forVar, writer.forVar))
    writer.write_module_forward(modName)
#}}}

def nn_logsoftmax(writer, modName, module): 
#{{{
    layerName = '_'.join(modName.split('.')[1:])
    moduleStr = str(module).split('(')[0]
    mod = '\t\tself.{} = nn.{}(dim={})'.format(layerName, moduleStr, module.dim)
    writer.toWrite['modules'].append(mod)
    
    writer.write_module_forward(modName)
#}}}

# custom modules
def residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op):
#{{{
    inputToBlock = writer.currIpChannels
    idx = writer.depBlk.instances.index(type(module))
    forVarBkp = writer.forVar

    # main path through residual
    if residual_branch is not None:
        opNode = '{}_main'.format(writer.forVar)
        forward = '\t\t{} = {}'.format(opNode, writer.forVar)
        writer.toWrite['forward'].append(forward)
        writer.forVar = opNode 
    
    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        if not any(x in n for x in writer.depBlk.dsLayers[idx]):
            main_branch(n, m, fullName, writer)
    
    writer.forVar = forVarBkp
    outputOfBlock = writer.currIpChannels

    if residual_branch is not None:
        # downsampling path if exists
        opNode1 = '{}_residual'.format(writer.forVar)
        forward = '\t\t{} = {}'.format(opNode1, writer.forVar)
        writer.toWrite['forward'].append(forward)
        forVarBkp = writer.forVar
        writer.forVar = opNode1
        
        for n,m in module.named_modules(): 
            fullName = "{}.{}".format(modName, n)
            if any(x in n for x in writer.depBlk.dsLayers[idx]):
                residual_branch(n, m, fullName, writer, inputToBlock, outputOfBlock)
        
        writer.forVar = forVarBkp
        
        if aggregation_op is not None:
            aggregation_op(writer, opNode, opNode1)
#}}}

def split_and_aggregate_backbone(writer, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs), 'For each branch a processing function must be provided - branches = {}, procFuns = {}'.format(len(branchStarts), len(branchProcs))

    inputToBlock = writer.currIpChannels
    forVarBkp = writer.forVar

    branchOpChannels = []
    opNodes = []
            
    for idx in range(len(branchStarts)):
        branchVar = "{}_{}".format(writer.forVar, idx)
        opNodes.append(branchVar)
        writer.toWrite['forward'].append("\t\t{} = {}".format(branchVar, writer.forVar))
        writer.forVar = branchVar
        writer.currIpChannels = inputToBlock

        inBranch = False
        for n,m in parentModule._modules.items(): 
            if n in branchStarts and not inBranch:
                inBranch = True
                branchStarts.pop(0)
            elif n in branchStarts and inBranch: 
                break
            
            if inBranch:
                fullName = "{}.{}".format(parentModName, n)
                branchProcs[idx](writer, fullName, m)
        
        branchOpChannels.append(writer.currIpChannels)
        writer.forVar = forVarBkp
	
    if aggregation_op is not None:
        aggregation_op(writer, opNodes, branchOpChannels)  
#}}}

def residual(writer, modName, module):
#{{{
    def main_branch(n, m, fullName, writer): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = writer.depBlk.instances.index(type(module))
            writer.addRelu = (n != writer.depBlk.convs[idx][-1])
            nn_conv2d(writer, fullName, m)

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, m)
            if writer.addRelu: 
                nn_relu(writer, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, writer, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            m.in_channels = ipToBlock
            m.out_channels = opOfBlock
            writer.currIpChannels = opOfBlock
            
            writer.write_module_desc(fullName, m)
            writer.write_module_forward(fullName)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
    #}}}

    def aggregation_op(writer, node1, node2):
    #{{{
        forward = '\t\t{} = F.relu({} + {}, inplace=True)'.format(writer.forVar, node1, node2)
        writer.toWrite['forward'].append(forward)
    #}}}
    
    residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op)
#}}}

def ofa_residual(writer, modName, module):
#{{{
    def main_branch(n, m, fullName, writer): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = writer.depBlk.instances.index(type(module))
            writer.addRelu = (n != writer.depBlk.convs[idx][-1])
            nn_conv2d(writer, fullName, m)

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, m)
            if writer.addRelu: 
                nn_relu(writer, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, writer, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.AvgPool2d): 
            nn_avgpool2d(writer, fullName, m)

        if isinstance(m, nn.Conv2d): 
            m.in_channels = ipToBlock
            m.out_channels = opOfBlock
            writer.currIpChannels = opOfBlock
            
            writer.write_module_desc(fullName, m)
            writer.write_module_forward(fullName)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
    #}}}

    def aggregation_op(writer, node1, node2):
    #{{{
        forward = '\t\t{} = F.relu({} + {}, inplace=True)'.format(writer.forVar, node1, node2)
        writer.toWrite['forward'].append(forward)
    #}}}
    
    residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op)
#}}}

def mb_conv(writer, modName, module):
#{{{
    def main_branch(n, m, fullName, writer): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = writer.depBlk.instances.index(type(module))
            writer.convIdx = writer.depBlk.convs[idx].index(n)
            nn_conv2d(writer, fullName, m, dw=(writer.convIdx==1))

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, m)
            if writer.convIdx == 0 or writer.convIdx == 1: 
                # nn_relu(writer, fullName, m)
                nn_relu6(writer, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, writer, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            m.in_channels = ipToBlock
            m.out_channels = opOfBlock
            writer.currIpChannels = opOfBlock
            
            writer.write_module_desc(fullName, m)
            writer.write_module_forward(fullName)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
    #}}}
    
    def aggregation_op(writer, node1, node2):
    #{{{
        forward = '\t\t{} = {} + {}'.format(writer.forVar, node1, node2)
        writer.toWrite['forward'].append(forward)
    #}}}
    
    idx = writer.depBlk.instances.index(type(module))
    midConv = writer.depBlk.convs[idx][1] 
    # stride = module._modules[midConv].stride[0]
    stride = dict(module.named_modules())[midConv].stride[0]
    
    residual = any(x in y for y in dict(module.named_modules()).keys() for x in writer.depBlk.dsLayers[idx]) 
    if residual:
        residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op)
    else:
        residual_backbone(writer, modName, module, main_branch, None, None)
#}}}

def fire(writer, modName, module): 
#{{{
    def basic(writer, fullName, module): 
    #{{{
        if isinstance(module, nn.Conv2d): 
            nn_conv2d(writer, fullName, module)

        if isinstance(module, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, module)
    #}}}

    def aggregation_op(writer, opNodes, branchOpChannels): 
    #{{{
        writer.currIpChannels = sum(branchOpChannels)
        nodes = "[{}]".format(','.join(opNodes))
        writer.toWrite['forward'].append("\t\t{} = torch.cat({}, 1)".format(writer.forVar, nodes))
        nn_relu(writer, None, None)
    #}}}
    
    inputToBlock = writer.currIpChannels
    idx = writer.depBlk.instances.index(type(module))
    convs = writer.depBlk.convs[idx]

    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if isinstance(m, nn.Conv2d): 
            if n == convs[0]:
                nn_conv2d(writer, fullName, m)
        
        if isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
        
        if isinstance(m, nn.ReLU): 
            nn_relu(writer, fullName, m)     
            split_and_aggregate_backbone(writer, modName, module, convs[1:], [basic,basic], aggregation_op)
            break
#}}}

def split_and_aggregate_backbone_new(writer, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs), 'For each branch a processing function must be provided - branches = {}, procFuns = {}'.format(len(branchStarts), len(branchProcs))

    inputToBlock = writer.currIpChannels
    forVarBkp = writer.forVar

    branchOpChannels = []
    opNodes = []
            
    for idx, branchName in enumerate(branchStarts):
        branchVar = "{}_{}".format(writer.forVar, idx)
        opNodes.append(branchVar)
        writer.toWrite['forward'].append("\t\t{} = {}".format(branchVar, writer.forVar))
        writer.forVar = branchVar
        writer.currIpChannels = inputToBlock

        fullName = "{}.{}".format(parentModName, branchName)
        branchProcs[idx](writer, fullName, dict(parentModule.named_modules())[branchName])
        
        branchOpChannels.append(writer.currIpChannels)
        writer.forVar = forVarBkp
	
    if aggregation_op is not None:
        aggregation_op(writer, opNodes, branchOpChannels)  
#}}}

def inception(writer, modName, module): 
#{{{
    def basic(writer, fullName, module): 
    #{{{
        for n,m in module.named_modules():
            if isinstance(m, nn.Conv2d): 
                nn_conv2d(writer, f"{fullName}.{n}", m)

            if isinstance(m, nn.BatchNorm2d): 
                nn_batchnorm2d(writer, f"{fullName}.{n}", m)
            
            if isinstance(m, nn.ReLU): 
                nn_relu(writer, f"{fullName}.{n}", m)

            if isinstance(m, nn.MaxPool2d): 
                nn_maxpool2d(writer, f"{fullName}.{n}", m)
    #}}}

    def aggregation_op(writer, opNodes, branchOpChannels): 
    #{{{
        writer.currIpChannels = sum(branchOpChannels)
        nodes = "[{}]".format(','.join(opNodes))
        writer.toWrite['forward'].append("\t\t{} = torch.cat({}, 1)".format(writer.forVar, nodes))
    #}}}
    
    writer.useCount = True
    inputToBlock = writer.currIpChannels
    idx = writer.depBlk.instances.index(type(module))
    convs = copy.deepcopy(writer.depBlk.convs[idx])
    split_and_aggregate_backbone_new(writer, modName, module, convs, [basic,basic,basic,basic], aggregation_op)
#}}}
