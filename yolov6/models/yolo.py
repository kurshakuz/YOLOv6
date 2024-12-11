#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers_lp = config.model.head_lp.num_layers
        num_layers_det = config.model.head_det.num_layers
        self.backbone, self.neck, self.detect_lp, self.detect_det = build_network(config, channels, num_classes, num_layers_lp, num_layers_det, fuse_ab=fuse_ab, distill_ns=distill_ns)

        # Init Detect head
        self.stride_lp = self.detect_lp.stride
        self.detect_lp.initialize_biases()

        self.stride_det = self.detect_det.stride
        self.detect_det.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x_lp = self.detect_lp(x)
        x_det = self.detect_det(x)
        return (x_lp, x_det) if export_mode is True else [(x_lp, x_det), featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect_lp.stride = fn(self.detect_lp.stride)
        self.detect_lp.grid = list(map(fn, self.detect_lp.grid))
        self.detect_det.stride = fn(self.detect_det.stride)
        self.detect_det.grid = list(map(fn, self.detect_det.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes_all, num_layers_lp, num_layers_det, fuse_ab=False, distill_ns=False):
    num_classes_lp = 1
    num_classes_det = num_classes_all - 1

    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl_lp = config.model.head_lp.use_dfl
    reg_max_lp = config.model.head_lp.reg_max
    use_dfl_det = config.model.head_det.use_dfl
    reg_max_det = config.model.head_det.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    if distill_ns:
        # from yolov6.models.heads.effidehead_distill_ns import Detect, DeHead, EffiDeHead
        # if num_layers != 3:
        #     LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
        #     exit()
        # HEAD = eval(config.model.head.type)
        # head_layers = HEAD(channels_list, 1, num_classes, reg_max=reg_max)
        # head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
        assert False, 'Not implemented yet.'

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect as DetectLP, DeHead, EffiDeHead
        from yolov6.models.heads.effidehead_fuseab_det import Detect as DetectDet, build_effidehead_layer

        anchors_init_lp = config.model.head_lp.anchors_init
        HEAD_LP = eval(config.model.head_lp.type)
        head_layers_lp = HEAD_LP(channels_list, 3, num_classes_lp, reg_max=reg_max_lp, num_layers=num_layers_lp)
        head_lp = DetectLP(num_classes_lp, anchors_init_lp, num_layers_lp, head_layers=head_layers_lp, use_dfl=use_dfl_lp)

        anchors_init_det = config.model.head_det.anchors_init
        head_layers_det = build_effidehead_layer(channels_list, 3, num_classes_det, reg_max=reg_max_det, num_layers=num_layers_det)
        head_det = DetectDet(num_classes_det, anchors_init_det, num_layers_det, head_layers=head_layers_det, use_dfl=use_dfl_det)

        return backbone, neck, head_lp, head_det

    else:
        from yolov6.models.effidehead import Detect as DetectLP, DeHead, EffiDeHead
        from yolov6.models.effidehead_det import Detect as DetectDet, build_effidehead_layer
        HEAD_LP = eval(config.model.head_lp.type)
        head_layers_lp = HEAD_LP(channels_list, 1, num_classes_lp, reg_max=reg_max_lp, num_layers=num_layers_lp)
        head_lp = DetectLP(num_classes_lp, num_layers_lp, head_layers=head_layers_lp, use_dfl=use_dfl_lp)

        head_layers_det = build_effidehead_layer(channels_list, 1, num_classes_det, reg_max=reg_max_det, num_layers=num_layers_det)
        head_det = DetectDet(num_classes_det, num_layers_det, head_layers=head_layers_det, use_dfl=use_dfl_det)

        return backbone, neck, head_lp, head_det

    return backbone, neck, head


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model
