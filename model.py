from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from grl import WarmStartGradientReverseLayer

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=False):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.cuda()#to(device=device)

    def forward(self, src_data, tar_data, aug_src_data = None, aug_tar_data = None, is_eval = False):
        return self.module(src_data, tar_data, aug_src_data, aug_tar_data, is_eval)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.cuda()#to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.cuda()#to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #There is no need to consisder initial way of ResNet if you load model from pretrained model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_mid=False, if_get_mid=False):
        if is_mid:
            x = self.layer4(x)
            return x
        if if_get_mid:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                mid = self.layer3(x)
            out = self.layer4(mid)
            return out, mid

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        mid = self.layer3(x)
        x = self.layer4(mid)
        
        '''
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 2048)
        '''

        return x

class LinearBottle(nn.Module):
    def __init__(self, input_dim=2048, bottle_dim=256):
        super(LinearBottle, self).__init__()
        self.input_dim = input_dim
        self.bottle = nn.Sequential(
                nn.Linear(input_dim, bottle_dim),
                nn.BatchNorm1d(bottle_dim),# if norm
                nn.ReLU(True)
                )

    def forward(self, x):
        '''
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, self.input_dim)
        '''
        out = self.bottle(x)

        return out

class ConvBottle(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBottle, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

class generalMHM(nn.Module):
    def __init__(self, backbone, heads, esm_bottle, esm_head, specific_proxy_classifiers, esm_proxy_classifier, discrimitor, specific_discriminators, weight_learner, branch_num, args):
        super(generalMHM, self).__init__()
        self.sharedNet = backbone
        self.esm_bottle = esm_bottle
        self.esm_head = esm_head
        self.esm_proxy_classifier = esm_proxy_classifier
        self.branch_num = branch_num
        self.weight_learner = weight_learner
        self.args = args
        self.grad_block = []
        self.fmap_block = []

        if self.branch_num == 2:
            self.hed1 = heads[0]
            self.hed2 = heads[1]
            self.heads = [self.hed1, self.hed2]

            self.spc1 = specific_proxy_classifiers[0]
            self.spc2 = specific_proxy_classifiers[1]
            self.spcs = [self.spc1, self.spc2]

            self.grl1 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500, auto_step=False)
            self.discrimitor1 = specific_discriminators[0]
            self.grl2 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500, auto_step=False)
            self.discrimitor2 = specific_discriminators[1]

            self.specific_grls = [self.grl1, self.grl2]
            self.specific_discrimitors = [self.discrimitor1, self.discrimitor2]

        if self.branch_num == 3:
            self.hed1 = heads[0]
            self.hed2 = heads[1]
            self.hed3 = heads[2]
            self.heads = [self.hed1, self.hed2, self.hed3]

            self.spc1 = specific_proxy_classifiers[0]
            self.spc2 = specific_proxy_classifiers[1]
            self.spc3 = specific_proxy_classifiers[2]
            self.spcs = [self.spc1, self.spc2, self.spc3]

            self.grl1 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500, auto_step=False)
            self.discrimitor1 = specific_discriminators[0]
            self.grl2 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500, auto_step=False)
            self.discrimitor2 = specific_discriminators[1]
            self.grl3 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=500, auto_step=False)
            self.discrimitor3 = specific_discriminators[2]

            self.specific_grls = [self.grl1, self.grl2, self.grl3]
            self.specific_discrimitors = [self.discrimitor1, self.discrimitor2, self.discrimitor3]

        if self.branch_num == 5:
            self.hed1 = heads[0]
            self.hed2 = heads[1]
            self.hed3 = heads[2]
            self.hed4 = heads[3]
            self.hed5 = heads[4]
            self.heads = [self.hed1, self.hed2, self.hed3, self.hed4, self.hed5]

            self.spc1 = specific_proxy_classifiers[0]
            self.spc2 = specific_proxy_classifiers[1]
            self.spc3 = specific_proxy_classifiers[2]
            self.spc4 = specific_proxy_classifiers[3]
            self.spc5 = specific_proxy_classifiers[4]
            self.spcs = [self.spc1, self.spc2, self.spc3, self.spc4, self.spc5]

            self.grl1 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=2000, auto_step=False)
            self.discrimitor1 = specific_discriminators[0]
            self.grl2 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=2000, auto_step=False)
            self.discrimitor2 = specific_discriminators[1]
            self.grl3 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=2000, auto_step=False)
            self.discrimitor3 = specific_discriminators[2]
            self.grl4 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=2000, auto_step=False)
            self.discrimitor4 = specific_discriminators[3]
            self.grl5 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=2000, auto_step=False)
            self.discrimitor5 = specific_discriminators[4]

            self.specific_grls = [self.grl1, self.grl2, self.grl3, self.grl4, self.grl5]
            self.specific_discrimitors = [self.discrimitor1, self.discrimitor2, self.discrimitor3, self.discrimitor4,
                                          self.discrimitor5]
            
            
        
    def block_out(self, x, idx=-1, need_branch=True, by_pass=False):
        branch_outs = []
        sharp_branch_outs = []
        branch_feas = []
        ext_focals = []
        esm_out = None
        esm_fea = None
        dis_out = None
        mid_f = self.sharedNet(x)
        mid_f = self.esm_bottle(mid_f)
        

        
        
        #dis_out = self.discrimitor(mid_f.detach())
        if need_branch:
            if idx >= 0:
                tf = mid_f#self.bottles[i](mid_f)
                to, tf, _ = self.heads[idx](tf, by_pass)
                branch_feas = tf
                branch_outs = to
                
            else:
                for i in range(0, self.branch_num):
                    tf = mid_f#self.bottles[i](mid_f)
                    to, tf, ext = self.heads[i](tf, by_pass)
                    ext_focals.append(ext)
                    branch_feas.append(tf)
                    branch_outs.append(to)
                    
                    

        esm_fea = mid_f#self.esm_bottle(mid_f)
        esm_out = self.esm_head(esm_fea)

        adv_f = None#self.grl(esm_fea)
        adv_out = None#self.discrimitor(adv_f)

        return branch_outs, esm_out, branch_feas, esm_fea, adv_out, ext_focals

    
    def forward(self, src_data, tar_data=None, aug_tar_data=None, is_eval = False, if_grl_step=True):

        if is_eval:
            branch_outs, esm_out, branch_feas, esm_fea, adv_out, ext_focals = self.block_out(src_data, by_pass=True)
            return branch_outs, esm_out, branch_feas, esm_fea
        
        main_src_preds = []
        esm_src_preds = []
        main_src_features = []
        esm_src_features = []
        main_dis_outs = []
        src_ext_focals = []
        
        for i in range(0, len(src_data)):
            branch_outs, esm_out, branch_feas, esm_f, dis_out, ext = self.block_out(src_data[i])
            src_ext_focals.append(ext[i])

            main_src_preds.append(branch_outs)
            esm_src_preds.append(esm_out)
            main_src_features.append(branch_feas[i])
            esm_src_features.append(esm_f)
            main_dis_outs.append(dis_out)
        
        main_tar_preds, esm_tar_preds, main_tar_feas, esm_tar_feas, tar_dis, _ = self.block_out(tar_data, by_pass=True)
        main_aug_tar_preds, esm_aug_tar_preds, main_aug_tar_feas, esm_aug_tar_feas, _, _ = self.block_out(aug_tar_data, by_pass=True)

        specific_src_proxy_outs = []
        esm_src_proxy_outs = []
        specific_tar_proxy_outs = []
        esm_tar_proxy_outs = None
        specific_aug_tar_proxy_outs = []
        esm_aug_tar_proxy_outs = None

        for i in range(0, len(src_data)):
            specific_src_proxy_outs.append(self.spcs[i](main_src_features[i].detach()))
            specific_tar_proxy_outs.append(self.spcs[i](main_tar_feas[i]))
            specific_aug_tar_proxy_outs.append(self.spcs[i](main_aug_tar_feas[i]))

        specific_src_advs = []
        specific_tar_advs = []
        specific_tar_advs_aug = []

        for i in range(0, len(src_data)):
            tadv = self.specific_grls[i](main_src_features[i])
            tadv = self.specific_discrimitors[i](tadv)
            specific_src_advs.append(tadv)

            tadv2 = self.specific_grls[i](main_tar_feas[i])
            tadv2 = self.specific_discrimitors[i](tadv2)
            specific_tar_advs.append(tadv2)

            tadv3 = self.specific_grls[i](main_aug_tar_feas[i])
            tadv3 = self.specific_discrimitors[i](tadv3)
            specific_tar_advs_aug.append(tadv3)
            

        tar_esm_weights = self.weight_learner(esm_tar_feas.detach())
        
        if if_grl_step:
            for i in range(0, len(src_data)):
                self.specific_grls[i].step()
        return main_src_preds, esm_src_preds, main_src_features, \
               main_tar_preds, esm_tar_preds, main_tar_feas, esm_tar_feas, tar_esm_weights, \
               main_aug_tar_preds, esm_aug_tar_preds, \
               main_dis_outs, tar_dis, specific_src_advs, specific_tar_advs, specific_tar_advs_aug, \
               specific_src_proxy_outs, specific_tar_proxy_outs, specific_aug_tar_proxy_outs, \
               esm_src_proxy_outs, esm_tar_proxy_outs, esm_aug_tar_proxy_outs


    def get_params(self, args, no_decay):
        params = [
            {'params': [p for n, p in self.sharedNet.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 0.1, "attribute": "no_warm"},
            {'params': [p for n, p in self.sharedNet.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 0.1, "attribute": "no_warm"}
        ]

        for i in range(0, self.branch_num):
            params += self.heads[i].get_params(args, no_decay)

        for i in range(0, self.branch_num):
            params.append({'params': [p for n, p in self.specific_discrimitors[i].named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "no_warm"})

            params.append({'params': [p for n, p in self.specific_discrimitors[i].named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "no_warm"})

        for i in range(0, self.branch_num):
            params.append({'params': [p for n, p in self.spcs[i].named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "no_warm"})

            params.append({'params': [p for n, p in self.spcs[i].named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "no_warm"})
        
        params.append({'params': [p for n, p in self.esm_bottle.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "atten"})
                
        params.append({'params': [p for n, p in self.esm_bottle.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "atten"})
    
        params.append({'params': [p for n, p in self.esm_head.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "no_warm"})
                
        params.append({'params': [p for n, p in self.esm_head.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "no_warm"})

        params.append({'params': [p for n, p in self.esm_proxy_classifier.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "atten"})

        params.append({'params': [p for n, p in self.esm_proxy_classifier.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "atten"})

        params.append({'params': [p for n, p in self.weight_learner.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1., "attribute": "atten"})

        params.append({'params': [p for n, p in self.weight_learner.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "atten"})

        return params

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, bottle_dim=256, if_drop=True, if_detach=False, if_zero=True):
        super(ImageClassifier, self).__init__()
        self.if_detach = if_detach
        self.bottle = nn.Sequential(
            nn.Linear(bottle_dim, bottle_dim//4),
            #nn.BatchNorm1d(bottle_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(bottle_dim//4, bottle_dim)
        )
        if if_drop:
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(bottle_dim, num_classes)
            )
            self.bottle[0].weight.data.normal_(0, 0.005)
            self.bottle[0].bias.data.fill_(0.1)
            if if_zero:
                self.bottle[2].weight.data.normal_(0, 0.0)
                self.bottle[2].bias.data.fill_(0.0)
            else:
                self.bottle[2].weight.data.normal_(0, 0.005)
                self.bottle[2].bias.data.fill_(0.1)
            self.head[1].weight.data.normal_(0, 0.01)
            self.head[1].bias.data.fill_(0.0)
        else:
            self.head = nn.Sequential(
                nn.Linear(bottle_dim, num_classes)
            )
            self.bottle[0].weight.data.normal_(0, 0.005)
            self.bottle[0].bias.data.fill_(0.1)
            if if_zero:
                self.bottle[2].weight.data.normal_(0, 0.0)
                self.bottle[2].bias.data.fill_(0.0)
            else:
                self.bottle[2].weight.data.normal_(0, 0.005)
                self.bottle[2].bias.data.fill_(0.1)
            self.head[0].weight.data.normal_(0, 0.01)
            self.head[0].bias.data.fill_(0.0)

    def forward(self, x, bypass=False):
        if bypass:
            return self.head(x), x, None
        f = self.bottle(x)
        out = self.head(x+f)

        if self.if_detach:
            return out, x + f, self.bottle(x.detach())
        return out, x+f, f

    def get_params(self, args, no_decay):
        params = [
            {'params': [p for n, p in self.bottle.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": args.res_weight,
             "attribute": "no_warm"},
            {'params': [p for n, p in self.bottle.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": args.res_weight, "attribute": "no_warm"},

            {'params': [p for n, p in self.head.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1.,
             "attribute": "no_warm"},
            {'params': [p for n, p in self.head.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "no_warm"}
        ]
        return params


class dImageClassifier(nn.Module):
    def __init__(self, num_classes, bottle_dim=256, if_drop=True, if_detach=False, if_zero=True):
        super(dImageClassifier, self).__init__()
        self.if_detach = if_detach
        self.bottle = nn.Sequential(
            nn.Linear(bottle_dim, bottle_dim//4),
            #nn.BatchNorm1d(bottle_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(bottle_dim//4, bottle_dim)
        )
        if if_drop:
            self.head = nn.Sequential(
                nn.Linear(bottle_dim, bottle_dim),
                nn.BatchNorm1d(bottle_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(bottle_dim, num_classes)
            )
            self.head[0].weight.data.normal_(0, 0.01)
            self.head[0].bias.data.fill_(0.05)
            self.head[4].weight.data.normal_(0, 0.01)
            self.head[4].bias.data.fill_(0.0)
            self.bottle[0].weight.data.normal_(0, 0.005)
            self.bottle[0].bias.data.fill_(0.1)
            if if_zero:
                self.bottle[2].weight.data.normal_(0, 0.0)
                self.bottle[2].bias.data.fill_(0.0)
            else:
                self.bottle[2].weight.data.normal_(0, 0.005)
                self.bottle[2].bias.data.fill_(0.1)
            self.head[1].weight.data.normal_(0, 0.01)
            self.head[1].bias.data.fill_(0.0)
        else:
            self.head = nn.Sequential(
                nn.Linear(bottle_dim, bottle_dim),
                nn.BatchNorm1d(bottle_dim),
                nn.ReLU(),
                nn.Linear(bottle_dim, num_classes)
            )
            self.head[0].weight.data.normal_(0, 0.01)
            self.head[0].bias.data.fill_(0.05)
            self.head[3].weight.data.normal_(0, 0.01)
            self.head[3].bias.data.fill_(0.0)
            self.bottle[0].weight.data.normal_(0, 0.005)
            self.bottle[0].bias.data.fill_(0.1)
            if if_zero:
                self.bottle[2].weight.data.normal_(0, 0.0)
                self.bottle[2].bias.data.fill_(0.0)
            else:
                self.bottle[2].weight.data.normal_(0, 0.005)
                self.bottle[2].bias.data.fill_(0.1)

    def forward(self, x, bypass=False):
        if bypass:
            return self.head(x), x
        f = self.bottle(x)
        out = self.head(x+f)

        if self.if_detach:
            return out, x + f, self.bottle(x.detach())
        return out, x+f, f

    def get_params(self, args, no_decay):
        params = [
            {'params': [p for n, p in self.bottle.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": args.res_weight,
             "attribute": "no_warm"},
            {'params': [p for n, p in self.bottle.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": args.res_weight, "attribute": "no_warm"},

            {'params': [p for n, p in self.head.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, "lr_mult": 1.,
             "attribute": "no_warm"},
            {'params': [p for n, p in self.head.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr_mult": 1., "attribute": "no_warm"}
        ]
        return params
        
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

#need backbone, bottleneck, number of source, number of classes, number of bottleneck_dim 
def get_model(args):
    if args.backbone =='resnet50':
        backbone = resnet50(args.pretrained)
    elif args.backbone =='resnet101':
        backbone = resnet101(args.pretrained)
    else:
        raise("unknown backbone: " + args.backbone)
    
    bottlenecks = []
    classifiers = []
    for i in range(0, len(args.src)+1):
        tb = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, args.bottleneck_dim),
            nn.BatchNorm1d(args.bottleneck_dim),
            nn.ReLU(),
        )
        tb[2].weight.data.normal_(0, 0.01)
        tb[2].bias.data.fill_(0.1)
        bottlenecks.append(tb)
        if args.name == 'nDomainNet':
            print("two layer branch")
            classifiers.append(
                dImageClassifier(args.num_classes, args.bottleneck_dim, if_drop=False, if_detach=args.if_detach,
                                if_zero=args.if_zero))
        else:
            classifiers.append(ImageClassifier(args.num_classes, args.bottleneck_dim, if_drop=False, if_detach = args.if_detach, if_zero=args.if_zero))
    
    esm_bottle = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, args.bottleneck_dim),
            nn.BatchNorm1d(args.bottleneck_dim),
            nn.ReLU(),
        )
    esm_bottle[2].weight.data.normal_(0, 0.005)
    esm_bottle[2].bias.data.fill_(0.1)

    if args.bottleneck == 'conv':
        esm_bottle = nn.Sequential(
            ConvBottle(2048, args.bottleneck_dim)
        )


    if args.name == 'nDomainNet':
        print("two layer esm-classifier")
        if args.is_drop:
            esm_classifier = nn.Sequential(
                nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            esm_classifier[0].weight.data.normal_(0, 0.01)
            esm_classifier[0].bias.data.fill_(0.05)
            esm_classifier[4].weight.data.normal_(0, 0.01)
            esm_classifier[4].bias.data.fill_(0.0)
        else:
            esm_classifier = nn.Sequential(
                nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
                #nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU(),
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            esm_classifier[0].weight.data.normal_(0, 0.01)
            esm_classifier[0].bias.data.fill_(0.05)
            esm_classifier[2].weight.data.normal_(0, 0.01)
            esm_classifier[2].bias.data.fill_(0.0)
    else:
        if args.is_drop:
            esm_classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            esm_classifier[1].weight.data.normal_(0, 0.01)
            esm_classifier[1].bias.data.fill_(0.0)
        else:
            esm_classifier = nn.Sequential(
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            esm_classifier[0].weight.data.normal_(0, 0.01)
            esm_classifier[0].bias.data.fill_(0.0)

    specific_proxy_classifiers = []
    for i in range(0, len(args.src)):
        tp = nn.Sequential(
            #nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(args.bottleneck_dim, args.num_classes)
        )
        specific_proxy_classifiers.append(tp)
    esm_proxy_classifier = nn.Sequential(
        #nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
        #nn.ReLU(),
        #nn.Dropout(0.5),
        nn.Linear(args.bottleneck_dim, args.num_classes)
    )

    if args.is_drop:
        discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(args.bottleneck_dim, args.num_classes)
        )
        discriminator[1].weight.data.normal_(0, 0.01)
        discriminator[1].bias.data.fill_(0.0)
    else:
        discriminator = nn.Sequential(
                    nn.Linear(args.bottleneck_dim, args.num_classes)
                )
        discriminator[0].weight.data.normal_(0, 0.01)
        discriminator[0].bias.data.fill_(0.0)

    specific_discriminators = []
    for i in range(0, len(args.src)):
        if args.name == 'DomainNet':
            print("two layer discriminators are used")
            
            td = nn.Sequential(
                nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU(),
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            td[0].weight.data.normal_(0, 0.01)
            td[0].bias.data.fill_(0.05)
            td[3].weight.data.normal_(0, 0.01)
            td[3].bias.data.fill_(0.0)

        else:
            td = nn.Sequential(
                nn.Linear(args.bottleneck_dim, args.num_classes)
            )
            td[0].weight.data.normal_(0, 0.01)
            td[0].bias.data.fill_(0.0)
        specific_discriminators.append(td)
    weight_learner = nn.Sequential(
        nn.Linear(args.bottleneck_dim, args.bottleneck_dim),
        nn.ReLU(),
        nn.Linear(args.bottleneck_dim, len(args.src))
        )

    model = generalMHM(backbone, classifiers, esm_bottle, esm_classifier, specific_proxy_classifiers, esm_proxy_classifier, discriminator, specific_discriminators, weight_learner, len(args.src), args)
    return model
