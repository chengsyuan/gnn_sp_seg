from utils.conf import conf
from utils.logger import logger

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
import dgl
import torch as th



class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()

        seed = conf.init_seed
        torch.manual_seed(seed)
        logger.info("random seed is {}".format(seed))

        # 1. extractor features
        vgg16_bn_backbone = vgg16_bn(pretrained=True)
        layer_list = list(vgg16_bn_backbone.children())[0][:6]
        self.features = nn.Sequential(*layer_list)

        # 2. assign each superpixel a semantic vecotr

        # 3. deep neural graph

    def forward(self, x):
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