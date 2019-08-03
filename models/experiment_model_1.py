from torch.nn.functional import upsample_bilinear, upsample
from torch.nn.modules import UpsamplingBilinear2d

from utils.conf import conf
from utils.logger import logger

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, resnet34
import dgl
import torch as th



class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()

        seed = conf.init_seed
        torch.manual_seed(seed)
        logger.info("random seed is {}".format(seed))

        # 1. extractor features
        # vgg16_bn_backbone = vgg16_bn(pretrained=True)
        # layer_list = list(vgg16_bn_backbone.children())[0][:6]
        # self.features = nn.Sequential(*layer_list)

        resnet34_backbone = resnet34(pretrained=True)
        self.features = nn.Sequential(
            resnet34_backbone.conv1,        # / 2
            resnet34_backbone.bn1,
            resnet34_backbone.relu,
            resnet34_backbone.maxpool,      # / 2

            resnet34_backbone.layer1,

            UpsamplingBilinear2d(scale_factor = 4)
        )


        # 2. assign each superpixel a semantic vecotr

        # 3. deep neural graph

    def forward(self, x):
        # print(x.size())                   # torch.Size([1, 3, 224, 224])
        # print(self.features(x).size())    # torch.Size([1, 64, 224, 224])

        fea = self.features(x)

        pass

    def build_graph(self, superpixel_labels):
        """

        :param superpixel_labels: [h, w] numpy array
        :return:
        """
        g = dgl.DGLGraph()
        g.add_nodes(len(superpixel_labels))

        start_list, end_list = [], []




        g.add_edges(start_list, end_list)
        g.ndata['h'] = th.randn(5, 3)  # assign feature  to each node
        g.edata['h'] = th.randn(4, 4)  # assign feature to each edge

        return g