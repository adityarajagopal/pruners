import sys

import torch.nn as nn
from itertools import chain
from abc import ABC,abstractmethod 

class DependencyCalculator(ABC):
#{{{
    def __init__(self):
        pass

    # three methods below used in dependency calculation for pruning
    @abstractmethod
    def dependent_conv(self, layerName, convs): 
        pass

    @abstractmethod
    def internal_dependency(self, module, mType, convs, ds):
        pass
    
    @abstractmethod
    def external_dependency(self, module, mType, convs, ds):
        pass
    
    # methods below used for incremental pruning percentage calculation
    @abstractmethod
    def get_internal_connections(self, name, module, convs, ds): 
        """
        Returns a dictionary where keys are each layer within the module
        and values a list with the names of the layers it is connected to     
        """
        pass
    
    @abstractmethod
    def get_interface_layers(self, name, module, convs, ds): 
        """
        Returns the layers that are directly connected to the input of the module
        """
        pass
#}}}

class Basic(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """Basic conv itself is the dependent conv"""
        return layerName
    
    def internal_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have internal dependencies"""
        return False,None
    
    def external_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have external dependencies"""
        return False,None

    def get_internal_connections(self, name, module, convs, ds): 
        """Basic conv is just connected to the next module, has no internal connectivity"""
        return {name: []}
    
    def get_interface_layers(self, name, module, convs, ds): 
        """Basic conv is just connected to the next module, has no internal connectivity"""
        return [(name, module.groups)] 
#}}}

class Linear(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """No convs inside"""
        return layerName
    
    def internal_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have internal dependencies"""
        return False,None
    
    def external_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have external dependencies"""
        return False,None

    def get_internal_connections(self, name, module, convs, ds): 
        """Basic conv is just connected to the next module, has no internal connectivity"""
        return {name: []}
    
    def get_interface_layers(self, name, module, convs, ds): 
        """Basic conv is just connected to the next module, has no internal connectivity"""
        return [(name, None)] 
#}}}

class Residual(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """dependent convolution is only the last convolution"""
        return "{}.{}".format(layerName, convs[-1])

    def internal_dependency(self, module, mType, convs, ds):
        """Residual blocks don't have internal dependencies"""
        return False,None

    def external_dependency(self, module, mType, convs, ds): 
    #{{{
        """
        Checks if module has a downsampling layer or not
        Some modules implement downsampling as just a nn.Sequential that's empty, so checks deeper to see if there is actually a conv inside
        Returns whether dependency exists and list of dependent layers
        """
        dependentLayers = [convs[-1]] 
        childrenAreDsLayers = [(c in ds) for c in list(module._modules.keys())]
        if any(childrenAreDsLayers):
            #check if empty sequential
            idx = childrenAreDsLayers.index(True)
            layerName = list(module._modules.keys())[idx]
            return not DependencyBlock.check_children(module._modules[layerName], [nn.Conv2d]), dependentLayers 
        else:
            return True, dependentLayers
    #}}}
    
    def get_internal_connections(self, name, module, convs, ds): 
    #{{{
        nextLayers = {}
        for n,m in module.named_modules(): 
            if isinstance(m, nn.Conv2d): 
                _n = "{}.{}".format(name,n)
                if n in convs:
                    idx = convs.index(n)
                    if idx != len(convs)-1:
                        nextConv = "{}.{}".format(name, convs[idx+1])
                        # groups = module._modules[convs[idx+1]].groups
                        groups = dict(module.named_modules())[convs[idx+1]].groups
                        nextLayers[_n] = [(nextConv, groups)]
                    else:
                        nextLayers[_n] = []
                
                elif any(x in n for x in ds): 
                    idx = [x in n for x in ds].index(True)
                    if idx != len(ds)-1:
                        nextConv = "{}.{}".format(name, ds[idx+1])
                        # groups = module._modules[ds[idx+1]].groups
                        groups = dict(module.named_modules())[ds[idx+1]].groups
                        nextLayers[_n] = [(nextConv, groups)]
                
                else:
                    print("ERROR : Detected conv in residual block that is not part of either main branch or residual branch")
                    sys.exit()
        
        return nextLayers
    #}}}
   
    def get_interface_layers(self, name, module, convs, ds): 
    #{{{
        # interfaceLayers = [("{}.{}".format(name, convs[0]), module._modules[convs[0]].groups)]
        interfaceLayers = [("{}.{}".format(name, convs[0]), dict(module.named_modules())[convs[0]].groups)]
        
        # only want first ds layer as this is the interface layer
        # assumption ds layers are listed in order --> ds[0]
        for n,m in module.named_modules(): 
            if n == ds[0] and DependencyBlock.check_children(m, [nn.Conv2d]): 
                if isinstance(m, nn.Conv2d): 
                    lName = n
                    groups = m.groups
                else:
                    idx = [isinstance(m, nn.Conv2d) for k,m in m._modules.items()].index(True)
                    subName = list(m._modules.keys())[idx]
                    lName = "{}.{}".format(n,subName)
                    groups = list(m._modules.values())[idx].groups
                interfaceLayers.append(("{}.{}".format(name, lName), groups))
        return interfaceLayers
    #}}}
#}}}

class MBConv(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """conv3 is the externally dependent conv with mb_conv blocks"""
        return "{}.{}".format(layerName, convs[2])

    def internal_dependency(self, module, mType, convs, ds):
        """convs 1 and 2 (dw convs) are internally dependent in mb_convs blocks"""
        return True, [convs[0], convs[1]]

    # def external_dependency(self, module, mType, convs, ds): 
    # #{{{
    #     """
    #     If no downsampling layer exists, then dependency exists
    #     If downsampling layer exists and it is of instance nn.conv2d, then no dependency exists
    #     If downsampling layer exists and does not have an nn.conv2d, if stide of conv2 is 1 dependency exists otherwise no
    #     Returns whether dependency exists and list of dependent layers
    #     """
    #     depLayers = [convs[2]]
    #     
    #     childrenAreDsLayers = [(c in ds) for c in list(module._modules.keys())]
    #     if any(childrenAreDsLayers):
    #         #check if empty sequential
    #         idx = childrenAreDsLayers.index(True)
    #         layerName = list(module._modules.keys())[idx]

    #         if DependencyBlock.check_children(module._modules[layerName], [nn.Conv2d]):
    #             return False, depLayers 
    #         else:
    #             if module._modules[convs[1]].stride[0] == 1:
    #                 return True, depLayers 
    #             else:
    #                 return False, depLayers
    #     else:
    #         return True, depLayers 
    # #}}}
    
    def external_dependency(self, module, mType, convs, ds): 
    #{{{
        """
        If no downsampling layer exists, then dependency exists
        If downsampling layer exists and it is of instance nn.conv2d, then no dependency exists
        If downsampling layer exists and does not have an nn.conv2d, if stide of conv2 is 1 dependency exists otherwise no
        Returns whether dependency exists and list of dependent layers
        """
        depLayers = [convs[2]]
        
        childrenAreDsLayers = [(c in ds) for c in list(module._modules.keys())]
        if any(childrenAreDsLayers):
            #check if empty sequential
            idx = childrenAreDsLayers.index(True)
            layerName = list(module._modules.keys())[idx]

            if DependencyBlock.check_children(module._modules[layerName], [nn.Conv2d]):
                return False, depLayers 
            else:
                return True, depLayers 
                # if module._modules[convs[1]].stride[0] == 1:
                #     return True, depLayers 
                # else:
                #     return False, depLayers
        else:
            # return True, depLayers 
            return False, depLayers 
    #}}}
    
    def get_internal_connections(self, name, module, convs, ds): 
    #{{{
        nextLayers = {}
        for n,m in module.named_modules(): 
            if isinstance(m, nn.Conv2d): 
                _n = "{}.{}".format(name,n)
                if n in convs:
                    idx = convs.index(n)
                    if idx != len(convs)-1:
                        nextConv = "{}.{}".format(name, convs[idx+1])
                        groups = dict(module.named_modules())[convs[idx+1]].groups
                        nextLayers[_n] = [(nextConv, groups)]
                    else:
                        nextLayers[_n] = []
                
                elif any(x in n for x in ds): 
                    idx = [x in n for x in ds].index(True)
                    if idx != len(ds)-1:
                        nextConv = "{}.{}".format(name, ds[idx+1])
                        groups = module._modules[ds[idx+1]].groups
                        nextLayers[_n] = [(nextConv, groups)]
                
                else:
                    print("ERROR : Detected conv in residual block that is not part of either main branch or residual branch")
                    sys.exit()
        
        return nextLayers
    #}}}
   
    def get_interface_layers(self, name, module, convs, ds): 
    #{{{
        interfaceLayers = [("{}.{}".format(name, convs[0]), dict(module.named_modules())[convs[0]].groups)]
        
        # only want first ds layer as this is the interface layer
        # assumption ds layers are listed in order --> ds[0]
        for n,m in module.named_modules(): 
            if n == ds[0] and DependencyBlock.check_children(m, [nn.Conv2d]): 
                if isinstance(m, nn.Conv2d): 
                    lName = n
                    groups = m.groups
                else:
                    idx = [isinstance(m, nn.Conv2d) for k,m in m._modules.items()].index(True)
                    subName = list(m._modules.keys())[idx]
                    lName = "{}.{}".format(n,subName)
                    groups = list(m._modules.values())[idx].groups
                interfaceLayers.append(("{}.{}".format(name, lName), groups))
        return interfaceLayers
    #}}}
#}}}

class Fire(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        return layerName 

    def internal_dependency(self, module, mType, convs, ds):
        return False, None 

    def external_dependency(self, module, mType, convs, ds): 
        return False, None 
    
    def get_internal_connections(self, name, module, convs, ds): 
    #{{{
        #TODO: make internal connection more about branches rather than fire specifically
        nextLayers = {}
        for n,m in module.named_modules(): 
            if isinstance(m, nn.Conv2d): 
                _n = "{}.{}".format(name,n)
                if n == convs[0]: 
                    nextLayers[_n] = []
                    for i in range(2): 
                        nextConv = "{}.{}".format(name, convs[i+1])
                        groups = module._modules[convs[i+1]].groups
                        nextLayers[_n].append((nextConv, groups))
                elif n == convs[1] or n == convs[2]:
                    nextLayers[_n] = []
                else:
                    print("ERROR : Detected conv in fire block that is not part of 3 convs declared")
                    sys.exit()
        return nextLayers
    #}}}
   
    def get_interface_layers(self, name, module, convs, ds): 
    #{{{
        interfaceLayers = [("{}.{}".format(name, convs[0]), module._modules[convs[0]].groups)]
        return interfaceLayers
    #}}}
#}}}

class Inception(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        return layerName 

    def internal_dependency(self, module, mType, convs, ds):
        return False, None 

    def external_dependency(self, module, mType, convs, ds): 
        return False, None 
    
    def get_internal_connections(self, name, module, branches, ds): 
    #{{{
        nextLayers = {}
        branchConvs = {branch:[] for branch in branches}
        for branch in branches: 
            for n,m in module.named_modules(): 
                if isinstance(m, nn.Conv2d) and branch in n:
                    branchConvs[branch].append((f"{name}.{n}", m.groups))
        
        for branch, convs in branchConvs.items(): 
            if len(convs) == 1: 
                nextLayers[convs[0][0]] = [] 
            else:
                for i,conv in enumerate(convs): 
                    nextLayers[conv[0]] = [convs[i+1]] if i < len(convs)-1 else []
        return nextLayers
    #}}}
   
    def get_interface_layers(self, name, module, branches, ds): 
    #{{{
        interfaceLayers = []
        seenBranches = []
        for n,m in module.named_modules(): 
            if isinstance(m, nn.Conv2d): 
                inBranch = [x in n for x in branches]
                if any(inBranch):
                    branch = branches[inBranch.index(True)]
                    # ensures only first conv of each branch returned
                    if branch not in seenBranches: 
                        interfaceLayers.append((f"{name}.{n}", m.groups))
                        seenBranches.append(branch)
        return interfaceLayers
    #}}}
#}}}

class DependencyBlock(object):
    def __init__(self, model):
    #{{{
        self.model = model
        
        try:
            self.types = self.dependentLayers['type']
            self.instances = self.dependentLayers['instance']
            self.convs = self.dependentLayers['conv']
            self.dsLayers = self.dependentLayers['downsample']
        except AttributeError: 
            print("Instantiating dependency block without decorators on model or for class without dependencies")

        if hasattr(self, 'depCalcs'):
            self.depCalcs['basic'] = Basic()
            self.depCalcs['linear'] = Linear()
        else: 
            self.depCalcs = {'basic': Basic(), 'linear': Linear()}
        
        self.linkedModules, linkedModulesWithFc = self.create_modules_graph()
        self.linkedConvAndFc = self.create_layer_graph(linkedModulesWithFc)
    #}}}
    
    @classmethod
    def update_block_names(cls, blockInst, **kwargs):
    #{{{
        lType = kwargs['lType']
        convs = None
        if 'convs' in kwargs.keys(): 
            convs = kwargs['convs']
        ds = None
        if 'downsampling' in kwargs.keys():
            ds = kwargs['downsampling']
        
        if hasattr(cls, 'dependentLayers'):
            cls.dependentLayers['instance'].append(blockInst)
            cls.dependentLayers['type'].append(lType)
            cls.dependentLayers['conv'].append(convs)
            cls.dependentLayers['downsample'].append(ds)
        else:
            setattr(cls, 'dependentLayers', {'type':[lType], 'instance':[blockInst], 'conv':[convs], 'downsample':[ds]})
    #}}}

    @classmethod
    def skip_layers(cls, blockInst, **kwargs): 
    #{{{
        if hasattr(cls, 'ignore'):
            cls.ignore.update({blockInst : ['module.{}'.format(x) for x in kwargs['convs']]})
        else:
            cls.ignore = {blockInst : ['module.{}'.format(x) for x in kwargs['convs']]}
    #}}}

    @classmethod 
    def register_dependency_calculator(cls, blockName, calcFunc):
    #{{{
        if hasattr(cls, 'depCalcs'): 
            cls.depCalcs[blockName] = calcFunc
        else: 
            setattr(cls, 'depCalcs', {blockName: calcFunc})
    #}}}
    
    @classmethod
    def check_children(cls, module, instances): 
    #{{{
        """Checks if module has any children that are of type in list instances""" 
        check = []
        for m in module.modules():
            check += [any(isinstance(m,inst) for inst in instances)]
        return any(check)
    #}}}

    @classmethod
    def check_inst(cls, module, instances): 
        """Checks if module is of one of the types in list instances"""
        return any(isinstance(module, inst) for inst in instances)
    
    def get_convs_and_ds(self, mName, mType, m): 
    #{{{
        if isinstance(m, nn.Conv2d):
            return [mName], [] 
        elif isinstance(m, nn.Linear): 
            return [], []
        else:
            convs = self.convs[self.types.index(mType)]
            ds = self.dsLayers[self.types.index(mType)]
            return convs, ds
    #}}}

    def create_modules_graph(self): 
    #{{{
        """
        Returns a list which has order of modules which have an instance of module in instances in the entire network
        eg. conv1 -> module1(which has as an mb_conv) -> conv2 -> module2 ...    
        """
        linkedConvModules = []
        linkedModules = []
            
        parentModule = None
        for n,m in self.model.named_modules(): 
            if DependencyBlock.check_inst(m, self.instances):
                parentModule = n
                idx = self.instances.index(type(m))
                mType = self.types[idx]
                linkedConvModules.append((mType, n))
                linkedModules.append((n, mType, m))

            elif isinstance(m, nn.Conv2d):
                if parentModule is None or parentModule not in n:
                    linkedConvModules.append(('basic', n))
                    linkedModules.append((n, 'basic', m))

            elif isinstance(m, nn.Linear): 
                if parentModule is None or parentModule not in n:
                    linkedModules.append((n, 'linear', m))

        return linkedConvModules, linkedModules
    #}}}
    
    def create_layer_graph(self, linkedModulesWithFc): 
    #{{{
        """
        Returns a list which has connectivity between all convs and FCs in the network 
        """
        linkedLayers = {}
        self.linkedConvs = {}
        for i in range(len(linkedModulesWithFc)-1):
            # get current module details
            mName, mType, m = linkedModulesWithFc[i]
            convs, ds = self.get_convs_and_ds(mName, mType, m) 
            
            # get next module details
            mNextName, mNextType, mNext = linkedModulesWithFc[i+1]
            convsNext, dsNext = self.get_convs_and_ds(mNextName, mNextType, mNext) 
            
            connected = self.depCalcs[mType].get_internal_connections(mName, m, convs, ds)
            connectedTo = self.depCalcs[mNextType].get_interface_layers(mNextName, mNext, convsNext, dsNext)
            
            for k,v in connected.items(): 
                if v == []: 
                    connected[k] = connectedTo
            
            linkedLayers.update(connected)
            if convs != []:
                self.linkedConvs.update(connected)
        
        return linkedLayers
    #}}}

    def get_dependencies(self):
    #{{{
        intDeps = []
        extDeps = []
        tmpDeps = []
        for n,m in self.model.named_modules(): 
            if DependencyBlock.check_inst(m, self.instances):
                idx = self.instances.index(type(m))
                mType = self.types[idx]
                convs = self.convs[idx]
                ds = self.dsLayers[idx]
                
                try:
                    internallyDep, depLayers = self.depCalcs[mType].internal_dependency(m, mType, convs, ds)
                except KeyError: 
                    print("CRITICAL WARNING : Dependency Calculator not defined for layer ({}). Assumed that no dependency exists".format(type(m)))
                    continue
                
                if internallyDep: 
                    intDeps.append(["{}.{}".format(n,x) for x in depLayers])
                
                externallyDep, depLayers = self.depCalcs[mType].external_dependency(m, mType, convs, ds)
                if externallyDep: 
                    depLayers = ["{}.{}".format(n,x) for x in depLayers]

                    if len(tmpDeps) != 0: 
                        tmpDeps += depLayers
                    else: 
                        bType,name = zip(*self.linkedModules)
                        idx = name.index(n)
                        prevType = bType[idx-1]
                        prev = self.depCalcs[prevType].dependent_conv(name[idx-1], convs)
                        tmpDeps = [prev,*depLayers]
                else: 
                    if len(tmpDeps) != 0:
                        extDeps.append(tmpDeps)
                    tmpDeps = []
            
            elif isinstance(m, nn.Conv2d): 
                if m.in_channels == m.groups: 
                    layers = dict(self.model.named_modules())
                    layerNames = list(layers.keys())
                    prevLayerName = [k for k,v in self.linkedConvs.items() if v[0][0] == n][0]
                    extDeps.append([prevLayerName, n])
        
        if len(tmpDeps) != 0: 
            extDeps.append(tmpDeps)

        return intDeps,extDeps
    #}}}

