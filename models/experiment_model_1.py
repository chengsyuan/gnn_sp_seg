from torch.nn.functional import upsample_bilinear, upsample
from torch.nn.modules import UpsamplingBilinear2d, upsampling

from utils.conf import conf
from utils.logger import logger

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, resnet34
import dgl
import torch.nn.functional as F
import dgl.function as fn

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(64, 64, F.relu)
        self.gcn2 = GCN(64, 32, F.relu)
        self.gcn3 = GCN(32, 21, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        x = self.gcn3(g, x)
        return x



class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()

        seed = conf.init_seed
        torch.manual_seed(seed)
        logger.info("random seed is {}".format(seed))

        # vgg16_bn_backbone = vgg16_bn(pretrained=True)
        # layer_list = list(vgg16_bn_backbone.children())[0][:6]
        # self.features = nn.Sequential(*layer_list)

        resnet34_backbone = resnet34(pretrained=True)
        self.features = nn.Sequential(
            resnet34_backbone.conv1,  # / 2
            resnet34_backbone.bn1,
            resnet34_backbone.relu,
            resnet34_backbone.maxpool,  # / 2

            resnet34_backbone.layer1,
        )

        self.net = Net()

    def forward(self, image, labels, edges, sp_size):
        labels = labels.squeeze(0).unsqueeze(2)  # [1,h,w] -> [h,w,1]
        h, w, _ = labels.size()

        fea = self.features(image)
        fea = F.upsample_bilinear(fea, (h, w)).squeeze(0). \
            permute([1, 2, 0]).unsqueeze(2) # [h,w,1,c]

        node_feature = self.get_node_feature(fea, labels, h, w, sp_size, margin=110)

        logger.debug('fea {}'.format(fea.size()))
        logger.debug('labels {}'.format(labels.size()))
        logger.debug('node_feature {}'.format(node_feature.size()))

        g = self.build_graph(edges, sp_size)

        logger.debug('graph {}'.format(g))

        logits = self.net(g, node_feature) # [sp_size, 64] -> [sp_size, 21]
        return logits

    def get_node_feature(self, feature, labels, h, w, sp_size, margin=100):
        # get one hot mask for each superpixel
        one_hot = torch.zeros(h, w, sp_size).cuda() \
            .scatter_(2, labels, 1).unsqueeze(3)

        node_feature = torch.zeros((sp_size, 64), dtype=torch.float).cuda()

        # cao 这个代码我想了一下午
        for st in range(0, sp_size, margin):
            ed = min(st + margin, sp_size)
            t = feature * one_hot[:, :, st:ed, :]
            t = t.mean(0).mean(0)

            node_feature[st:ed] = t
            # node_feature

        # for i in range(sp_size):
        #     cur_sp_mask = one_hot[:, :, i]
        #     f = cur_sp_mask * fea
        #     node_feature[i, :] = f.mean(0).mean(0)
        #     print(i)
        #
        # print(node_feature.detach().numpy())
        return node_feature

    def build_graph(self, edges, sp_size):
        g = dgl.DGLGraph()

        g.add_nodes(sp_size)
        start_list, end_list = edges[0,:], edges[1,:]
        g.add_edges(start_list, end_list)

        return g
