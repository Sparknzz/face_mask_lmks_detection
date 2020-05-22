import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from models.backbones import *

class ClassHead(nn.Module):
    def __init__(self, class_num, inchannels, num_anchors):
        # num_cls including neg, face, mask
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.class_num = class_num
        # bit different here, compare to faster rcnn anchor mechanism, the rcnn doesn't have anchor anymore, 
        # but for single stage model, it deponds on anchors number, while rcnn consider the num class. 4 * 2 sigmoid or 4 * 3 softmax
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * class_num, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.class_num)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        # as cls head has 3 number, mask face neg. for bbox head, each anchor will have a 4 coordinates. but why it doesn't need to consider which cls to regress.
        # cos only pos need to be regress, so only one cls here. but now we got 2 different pos cls, need to be multiply 2.
        # 4 * 4， means each anchor is only correspond to one cls not two
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)

class RetinaFace(nn.Module):
    def __init__(self, class_num=3, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet':
            backbone = MobileNetV1()
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['name'] == 'resnet18':
            import torchvision.models as models
            backbone = models.resnet18(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers']) # get output of each layers

        in_channels_stage2 = cfg['in_channel']

        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg['out_channel']
        
        # in_channel 256 512 1024
        self.fpn = FPN(in_channels_list, out_channels)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(class_num, fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, class_num, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(class_num, inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications)
        else:
            output = (bbox_regressions, F.sigmoid(classifications))
        return output