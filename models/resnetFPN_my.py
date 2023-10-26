import torch.nn as nn
import math, pdb
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from models.nonlocal_helper import Nonlocal
from models.blocks import *

import modules.utils as lutils

import yaml
from easydict import EasyDict
from .backbones import AVA_backbone



logger = lutils.get_logger(__name__)

### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0,padding,padding), bias=bias)


def conv1x1(in_channel, out_channel):
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)


from modeling_finetune import vit_base_patch16_224

class ResNetFPN(nn.Module):

    def __init__(self, block, args):
        
        self.inplanes = 64
        super(ResNetFPN, self).__init__()
    
    
        if args.MODEL_TYPE.startswith('SlowFast'):
            with open('configs/ROAD.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            opt = EasyDict(config)
            print(opt)
            self.inplanes = 64
            super(ResNetFPN, self).__init__()
            self.backbone = AVA_backbone(opt)
    
        if args.MODEL_TYPE.startswith('videomae'):
            self.backbone = vit_base_patch16_224(all_frames=8, img_size=(512, 704)).half()
            checkpoint = torch.load('vit_b_k710_dl_from_giant.pth')['module']
            checkpoint = {k: v for k, v in checkpoint.items() if 'head' not in k}
            self.backbone.load_state_dict(checkpoint, strict=False)

        self.MODEL_TYPE = args.MODEL_TYPE
        #num_blocks = args.model_perms
        #non_local_inds = args.non_local_inds
        #model_3d_layers = args.model_3d_layers
        #self.num_blocks = num_blocks
        #self.non_local_inds = non_local_inds
        #self.model_3d_layers = model_3d_layers
        #self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
        #                       bias=False)
        #self.bn1 = nn.BatchNorm3d(64)
        #self.relu = nn.ReLU(inplace=True)
        #self.pool1 = nn.MaxPool3d(kernel_size=(
        #    1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        #self.layer_names = []
        #self.layer1 = self._make_layer(
        #    block, 64, num_blocks[0], temp_kernals=model_3d_layers[0], nl_inds=non_local_inds[0])
        #self.MODEL_TYPE = args.MODEL_TYPE
        
        if args.model_subtype in ['C2D','RCN','CLSTM','RCLSTM','CGRU','RCGRU']:
            self.pool2 = None
        else:
            self.pool2 = nn.MaxPool3d(kernel_size=(
                2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        #self.layer2 = self._make_layer(
        #    block, 128, num_blocks[1], stride=2, temp_kernals=model_3d_layers[1], nl_inds=non_local_inds[1])
        #self.layer3 = self._make_layer(
        #    block, 256, num_blocks[2], stride=2, temp_kernals=model_3d_layers[2], nl_inds=non_local_inds[2])
        #self.layer4 = self._make_layer(
        #    block, 512, num_blocks[3], stride=2, temp_kernals=model_3d_layers[3], nl_inds=non_local_inds[3])

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.MODEL_TYPE == 'SlowFast':
            self.conv6 = conv3x3(2304, 256, stride=2, padding=1)  # P6
            self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

            self.ego_lateral = conv3x3(512 * block.expansion,  256, stride=2, padding=0)
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            self.lateral_layer1 = conv1x1(2304, 256)
            self.lateral_layer2 = conv1x1(1152, 256)
            self.lateral_layer3 = conv1x1(576, 256)
            
            self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3
        elif self.MODEL_TYPE == 'videomae':
            self.conv_layer = torch.nn.ConvTranspose3d(in_channels=768,out_channels=768, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False).half()
            self.conv1 = nn.Conv3d(768, 768, kernel_size=1, bias=False).half()
            self.norm1 = nn.BatchNorm3d(768).half()
                
            self.conv2 = nn.Conv3d(768, 768, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False).half()
            self.norm2 = nn.BatchNorm3d(768).half()
            self.conv3 = nn.Conv3d(768, 768, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False).half() 
            
            self.conv4 = nn.Conv3d(768*2, 768*2, kernel_size=1, bias=False).half()
            self.norm3 = nn.BatchNorm3d(768*2).half()
            self.conv5 = nn.Conv3d(768*2, 768*2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False).half()
            self.norm4 = nn.BatchNorm3d(768*2).half()

            self.conv6 = conv3x3(2*768, 256, stride=2, padding=1).half()  # P6
            self.conv7 = conv3x3(256, 256, stride=2, padding=1).half()  # P7
            
            self.ego_lateral = conv3x3(512 * block.expansion,  256, stride=2, padding=0).half()
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)).half()

            self.lateral_layer1 = conv1x1(2*768, 256).half()
            self.lateral_layer2 = conv1x1(768, 256).half()
            self.lateral_layer3 = conv1x1(768, 256).half()

            self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1).half()  # P4
            self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1).half()  # P4
            self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1).half()  # P3
            
        else:
            self.conv6 = conv3x3(512 * block.expansion, 256, stride=2, padding=1)  # P6
            self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

            # self.ego_lateral = conv3x3(512 * block.expansion,  256, stride=2, padding=0)
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

            self.lateral_layer1 = conv1x1(512 * block.expansion, 256)
            self.lateral_layer2 = conv1x1(256 * block.expansion, 256)
            self.lateral_layer3 = conv1x1(128 * block.expansion, 256)
            
            self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
            self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if hasattr(m.bias, 'data'):
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _upsample(self, x, y):
        _, _, t, h, w = y.size()
        # print('spatial', x.shape, y.shape)
        x_upsampled = F.interpolate(x, [t, h, w], mode='nearest')

        return x_upsampled

    def _upsample_time(self, x):
        _,_,t, h, w = x.size()
        # print('time', x.shape)
        x_upsampled = F.interpolate(x, [t*2, h, w], mode='nearest')

        return x_upsampled

    def _make_layer(self, block, planes, num_blocks, stride=1, temp_kernals=[], nl_inds=[]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layer_names = []
        if 0 in temp_kernals:
            layers.append(block(self.inplanes, planes, stride,
                                temp_kernal=3, downsample=downsample))
        else:
            layers.append(block(self.inplanes, planes, stride,
                                temp_kernal=0, downsample=downsample))
        layer_names.append(0)
        self.inplanes = planes * block.expansion

        for i in range(1, num_blocks):
            if i in temp_kernals:
                layers.append(block(self.inplanes, planes, temp_kernal=3))
            else:
                layers.append(block(self.inplanes, planes, temp_kernal=1))
            layer_names.append(i)

            if i in nl_inds:
                nln = Nonlocal(
                    planes * block.expansion,
                    (planes * block.expansion) // 2,
                    [1, 2, 2],
                    instantiation='softmax',
                )
                layers.append(nln)

        self.layer_names.append(layer_names)

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.MODEL_TYPE.startswith('SlowFast'):
            ff = self.backbone(x)
            c3 = ff[0]
            c4 = ff[1]
            c5 = ff[2]
            print(c3.shape)
            print(c4.shape)
            print(c5.shape)
            p5 = self.lateral_layer1(c5)
            p5_upsampled = self._upsample(p5, c4)
            p5 = self.corr_layer1(p5)
            p4 = self.lateral_layer2(c4)
            p4 = p5_upsampled + p4
            p4_upsampled = self._upsample(p4, c3)
            p4 = self.corr_layer2(p4)
            p3 = self.lateral_layer3(c3)
            p3 = p4_upsampled + p3
            p3 = self.corr_layer3(p3)
            p6 = self.conv6(c5)
            p7 = self.conv7(F.relu(p6))
            features = [p3, p4, p5, p6, p7]
            ego_feat = self.avg_pool(p7)
            if self.pool2 is not None:
                for i in range(len(features)):
                    features[i] = self._upsample_time(features[i])
                ego_feat = self._upsample_time(ego_feat)
        elif self.MODEL_TYPE.startswith('videomae'):
            x = x.half()
            ff = self.backbone.get_features(x)
            ff = ff.permute((0,2,1))
            vit_output_featuremap = torch.reshape(ff,(ff.shape[0], ff.shape[1], 4, 32, 44))
            
            c3 = self.conv_layer(vit_output_featuremap)
            c3 = self.norm1(self.conv1(c3)).relu()
            c3 = self.norm2(self.conv2(c3)).relu()
            
            c4 = vit_output_featuremap
            c5 = self.conv3(c3)
            c5 = torch.cat([c4, c5], dim=1)
            
            c5 = self.norm3(self.conv4(c5)).relu()
            c5 = self.norm4(self.conv5(c5)).relu()
            
            p5 = self.lateral_layer1(c5) 
            p5_upsampled = self._upsample(p5, c4)
            p5 = self.corr_layer1(p5)
            p4 = self.lateral_layer2(c4)
            p4 = p5_upsampled + p4
            p4_upsampled = self._upsample(p4, c3)
            p4 = self.corr_layer2(p4)
            p3 = self.lateral_layer3(c3)
            p3 = p4_upsampled + p3
            p3 = self.corr_layer3(p3)
            p6 = self.conv6(c5)
            p7 = self.conv7(F.relu(p6))
            
            features = [p3.float(), p4.float(), p5.float(), p6.float(), p7.float()]
            if self.pool2 is not None:
                for i in range(len(features)):
                    features[i] = self._upsample_time(features[i])
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool1(x)
            x = self.layer1(x)
            if self.pool2 is not None:
                x = self.pool2(x)
            c3 = self.layer2(x)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            p5 = self.lateral_layer1(c5)
            p5_upsampled = self._upsample(p5, c4)
            p5 = self.corr_layer1(p5)
            p4 = self.lateral_layer2(c4)
            p4 = p5_upsampled + p4
            p4_upsampled = self._upsample(p4, c3)
            p4 = self.corr_layer2(p4)
            p3 = self.lateral_layer3(c3)
            p3 = p4_upsampled + p3
            p3 = self.corr_layer3(p3)
            p6 = self.conv6(c5)
            p7 = self.conv7(F.relu(p6))
            features = [p3, p4, p5, p6, p7]
            # ego_feat = self.avg_pool(p7)
            if self.pool2 is not None:
                for i in range(len(features)):
                    features[i] = self._upsample_time(features[i])
                # ego_feat = self._upsample_time(ego_feat)
        return features


    def identity_state_dict(self):
        state_dict = self.state_dict()
        logger.info('** Initilise identities **')
        for name, param in state_dict.items():
            if name.find('whh') > -1:
                logger.info('Set zeros for ' + name)
                p = param * 0.0
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    p = param.data * 0.0
                # pdb.set_trace()
                for i in range(p.size(0)):
                    p[i, i, :, :, :] = 1.0/(p.size(2)*p.size(3)*p.size(4))
                try:
                    state_dict[name].copy_(p)
                    logger.info('Init' + name + ' ==>>' + str(p.size))
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))


    def recurrent_conv_zero_state(self):
        state_dict = self.state_dict()
        print('**Initilise Recurrences for LSTMs or GRU **')
        for name, param in state_dict.items():
            if name.find('recurrent_conv') > -1:
                p = param * 0.0
                print('Set zeros for ' + name + str(torch.sum(torch.abs(param))))
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    p = param.data * 0.0
                try:
                    state_dict[name].copy_(p)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

                                       
    def load_my_state_dict(self, state_dict):
        logger.info('**LOAD STAE DICTIONARY FOR WEIGHT INITLISATIONS**')
        own_state = self.state_dict()
        update_name_dict = {}
        for b in range(len(self.non_local_inds)):
            count = -1
            for i in range(self.num_blocks[b]):
                count += 1
                sdlname = 'layer{:d}.{:d}'.format(b+1, i)
                oslname = 'layer{:d}.{:d}'.format(b+1, count)
                update_name_dict[oslname] = sdlname
                if i in self.non_local_inds[b]:
                    count += 1

        for own_name in own_state:
            state_name = own_name
            discard = False
            if own_name.startswith('layer'):
                sn = own_name.split('.')
                ln = sn[0]+'.'+sn[1]
                if ln in update_name_dict:
                    state_name = own_name.replace(ln, update_name_dict[ln])
                else:
                    discard = True
            # pdb.set_trace()
            if 'module.'+state_name in state_dict.keys():
                state_name = 'module.'+state_name 
            if state_name in state_dict.keys() and not discard:
                param = state_dict[state_name]
                own_size = own_state[own_name].size()
                param_size = param.size()
                match = True
                if len(param_size) != len(own_size) and len(own_size) > 1:
                    repeat_dim = own_size[2]
                    param = param.unsqueeze(2)
                    param = param.repeat(
                        1, 1, repeat_dim, 1, 1)/float(repeat_dim)
                else:
                    for i in range(len(param_size)):
                        if param_size[i] != own_size[i]:
                            match = False
                if match:
                    own_state[own_name].copy_(param)
                    logger.info(own_name + str(param_size) + ' ==>>' + str(own_size))
            else:
                logger.info('NAME IS NOT INPUT STATE DICT::>' + state_name)

def resnetfpn(args):
    model_type = args.MODEL_TYPE
    if model_type.startswith('I3D'):
        return ResNetFPN(BottleneckI3D, args)
    elif model_type.startswith('RCN'):
        return ResNetFPN(BottleneckRCN, args)
    elif model_type.startswith('2PD'):
        return ResNetFPN(Bottleneck2PD, args)
    elif model_type.startswith('C2D'):
        return ResNetFPN(BottleneckC2D, args)
    elif model_type.startswith('CLSTM'):
        return ResNetFPN(BottleneckCLSTM, args)
    elif model_type.startswith('RCLSTM'):
        return ResNetFPN(BottleneckRCLSTM, args)
    elif model_type.startswith('RCGRU'):
        return ResNetFPN(BottleneckRCGRU, args)
    elif model_type.startswith('SlowFast'):
        return ResNetFPN(BottleneckI3D, args)
    elif model_type.startswith('videomae'):
        return ResNetFPN(BottleneckI3D, args)
    else:
        raise RuntimeError('Define the model type correctly:: ' + model_type)
