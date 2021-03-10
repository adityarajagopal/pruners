import sys

import model_writers as writers
import dependencies as dependencies
import weight_transfer as weight_transfer

def check_kwargs(**kwargs): 
#{{{
    """Check that kwargs has atleast the lType argument, if not it is an invalid decorator"""
    if 'lType' in kwargs.keys(): 
        return 
    else: 
        print("ERROR : Decorator without lType argument declared. This is invalid")
        sys.exit()
#}}}

def residual(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.ResidualDependencyBlock.update_block_names(block, **kwargs)
        dependencies.ResidualDependencyBlock.register_dependency_calculator(block, kwargs['lType'],\
                dependencies.SEResidual())
        return block
    return decorator
#}}}

def se_residual(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.SEResidualDependencyBlock.update_block_names(block, **kwargs)
        dependencies.SEResidualDependencyBlock.register_dependency_calculator(block, kwargs['lType'],\
                dependencies.SEResidual())
        return block
    return decorator
#}}}

def ofa_residual(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Residual())
        writers.Writer.register_writer(kwargs['lType'], writers.ofa_residual)
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.residual)
        return block
    return decorator
#}}}
 
def ofa_global_average_pool(**kwargs):
#{{{
    def decorator(block): 
        writers.Writer.register_writer(block.__name__.lower(), writers.ofa_adaptiveavgpool2d)
        weight_transfer.WeightTransferUnit.register_transfer_func(block.__name__.lower(), weight_transfer.ofa_adaptiveavgpool2d)
        return block
    return decorator
#}}}

def mb_conv(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.MBConv())
        writers.Writer.register_writer(kwargs['lType'], writers.mb_conv)
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.mb_conv)
        return block
    return decorator
#}}}

def fire(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Fire())
        writers.Writer.register_writer(kwargs['lType'], writers.fire)
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.fire)
        return block
    return decorator
#}}}

def inception(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Inception())
        writers.Writer.register_writer(kwargs['lType'], writers.inception)
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.inception)
        return block
    return decorator
#}}}

def skip(**kwargs): 
#{{{
    def decorator(block): 
        dependencies.DependencyBlock.skip_layers(block, **kwargs)
        return block 
    return decorator
#}}}
