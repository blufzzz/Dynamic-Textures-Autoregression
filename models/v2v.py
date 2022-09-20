# from https://github.com/blufzzz/learnable-triangulation-pytorch/blob/master/mvn/models/v2v.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
from collections import OrderedDict
import sys
import inspect
import numpy as np


NORMALIZATION = 'batch_norm'
ACTIVATION = 'ReLU'

def normalization(out_planes):
    if NORMALIZATION == 'batch_norm':
        return nn.BatchNorm3d(out_planes, affine=False)
    elif NORMALIZATION == 'group_norm':
        return nn.GroupNorm(16, out_planes)
    elif NORMALIZATION == 'instance_norm':
        return nn.InstanceNorm3d(out_planes, affine=True)
    else:
        raise RuntimeError('Wrong NORMALIZATION')

def activation():
    if ACTIVATION == 'ReLU':
        return nn.ReLU(True)
    elif ACTIVATION == 'ELU':
        return nn.ELU()
    elif ACTIVATION == 'LeakyReLU':
        return nn.LeakyReLU()
    else:
        raise RuntimeError('Wrong ACTIVATION')



class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            normalization(out_planes),
            activation()
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            normalization(out_planes),
            activation(),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            normalization(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                normalization(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            normalization(out_planes),
            activation()
        )

    def forward(self, x):
        return self.block(x)


    
   
    
class EncoderDecorder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.skip_concat = config.skip_concat # use concatenation + convolution (line in U-Net)
        
        # see v2v.yaml for config
        downsampling_config = config.v2v_configuration.downsampling
        bottleneck_config = config.v2v_configuration.bottleneck
        upsampling_config = config.v2v_configuration.upsampling
        skip_block_type = config.v2v_configuration.skip_block_type
        
        assert len(downsampling_config) == len(upsampling_config), 'Full encoder-decoder symmetry is implied!'
        N_blocks = len(downsampling_config)

        self.downsampling_list = nn.ModuleList()
        self.bottleneck_list = nn.ModuleList()
        self.upsampling_list = nn.ModuleList()
        self.skip_connection_list = nn.ModuleList()
        self.skip_connection_order = []

        # creating downsampling path
        for i, module_config in enumerate(downsampling_config):

            self._append_module(self.downsampling_list, module_config)
            module_type = module_config['module_type']
            module_skip_connection = module_config['skip_connection'] \
                                     if 'skip_connection' in module_config else False

            # add skip-connection
            if module_skip_connection:
                # FULL encoder-decoder symmetry is implied!
                in_planes = module_config['params'][-1]
                out_planes = upsampling_config[-i-1]['params'][0]

                skip_con_block = getattr(sys.modules[__name__], skip_block_type)(in_planes, out_planes) 

                self.skip_connection_list.append(skip_con_block)
                # save layer number after which we obtain skip-connection
                self.skip_connection_order.append(i)
           
        # creating bottleneck path
        for i, module_config in enumerate(bottleneck_config):  
            self._append_module(self.bottleneck_list, module_config)
           
        # creating upsampling path
        for i, module_config in enumerate(upsampling_config):
            
            # check if skip connection is sent by downsampling branch
            try ????
            if downsampling_config[N_blocks-i-1]['skip_connection']:
                if self.skip_concat and module_config['module_type'] == 'Res3DBlock':
                    # change input planes
                    module_config['params'][0] = 2*module_config['params'][0]
            
            self._append_module(self.upsampling_list, module_config)


    def _append_module(self, modules_list, module_config):

            module_type = module_config['module_type']
            constructor = getattr(sys.modules[__name__], module_config['module_type'])

            # construct params dict                         
            model_params = {}
            # get arguments and omitting `self`
            arguments = inspect.getargspec(constructor.__init__).args[1:]
            for i, param in enumerate(module_config['params']): 
                # get arguments for __init__ function
                param_name = arguments[i]                            
                model_params[param_name] = param

            module = constructor(**model_params)
            modules_list.append(module)

    def forward(self, x):
        
        skip_connections = []
        
        # downsampling path
        for i, module in enumerate(self.downsampling_list):
            x = module(x)
            # means we need to save output of this layer as skip connection
            if i in self.skip_connection_order: 
                k = len(skip_connections) # number of the skip connection we creating
                skip_connection = self.skip_connection_list[k](x)
                skip_connections.append(skip_connection)
                    
        # bottleneck path
        for i, module in enumerate(self.bottleneck_list):
            x = module(x)
          
        skip_number = -1 # staring from the last skip_connection
        n_ups_blocks = len(self.upsampling_dict)
        
        # upsampling path
        for i, module in enumerate(self.upsampling_list):
            
            # adding skip-connection
            if i in self.skip_connection_order:
                skip_connection = skip_connections[skip_number]
                
                # interpolate if size does not match
                if x.shape[-3:] != skip_connection.shape[-3:]:
                    set_trace()
                    skip_connection = F.interpolate(skip_connection,
                                                    size=(x.shape[-3:]), 
                                                    mode='trilinear')
                    
                if self.skip_concat:
                    x = torch.cat([skip_connection, x], 1)
                else:
                    x = x + skip_connection
                skip_number -= 1
            
            x = module(x)

        return x                  

    

class V2VModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sigmoid = config.sigmoid
        self.input_channels = config.input_channels
        self.output_channels = self.input_channels
        
        self.skip_concat = config.skip_concat if hasattr(config, 'skip_concat') else False
        self.use_skip_connections = config.use_skip_connections

        # nasty hack to replace normalization\activation layers in the model
        global NORMALIZATION
        NORMALIZATION = config.normalization if hasattr(config,'normalization') else 'batch_norm'
        global ACTIVATION
        ACTIVATION = config.activation if hasattr(config,'activation') else 'ReLU'

        self.front_layers = nn.Sequential(
            normalization(self.input_channels),
            Basic3DBlock(self.input_channels, 16, 3),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder(config)

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, self.output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):

        # x - [bs, C, x,x,x]

        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)

        if self.sigmoid:
            return x.sigmoid()
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

                
           