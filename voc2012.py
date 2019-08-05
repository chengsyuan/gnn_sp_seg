import xmltodict
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import RandomResizedCrop, Normalize, ToTensor, CenterCrop

from utils import superpixel
from utils.conf import conf
from utils.logger import logger

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import torchvision

from utils.mytransformers import ToLabel, Relabel
from utils.superpixel import superpixel_edge_list, superpixel
import numpy as np

logger.info('torchvsion {}'.format(torchvision.__version__))

batch_size = conf.batch_size


class myVOC2012(Dataset):
    class_names = [
        'background',
        'plane',
        'bike',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'table',
        'dog',
        'horse',
        'motorbike',
        'person',
        'plant',
        'sheep',
        'sofa',
        'train',
        'monitor',
    ]

    def __init__(self):
        self.id2name = {}
        self.name2id = {}
        for idx, name in enumerate(self.class_names):
            self.id2name[idx] = name
            self.name2id[name] = idx

        self.train_dataset = \
            VOCSegmentation('./', image_set='train', )

        self.image_trans = transforms.Compose([
            ToTensor(),
        ])

        self.mask_trans = transforms.Compose([
            ToLabel(),
            Relabel(255, 21)  # change 255 to 21
        ])

        self.label_trans = transforms.Compose([
            ToTensor()
        ])

        self.y_trans = transforms.Compose([
            ToTensor()
        ])

    def __getitem__(self, index):
        """

        :param index:
        :return:
        image: [3, h ,w]
        mask: [h, w]
        labels: [h, w]
        edgelist: [2, len(sp)] len(sp) = numbers of unique superpixels
        """

        image, mask = self.train_dataset[index]

        labels, unique_ids = superpixel(image, debug=True)
        # print(labels)

        # print(111)
        edges = superpixel_edge_list(labels)
        # print(222)


        w, h = image.size

        y_gt = {}
        polygons = self.getpolygons(index)
        # print(index)
        for s1 in polygons:
            tag = s1['tag']
            label_id = self.name2id[tag]

            for s2 in s1['point']:
                x, y = int(s2['X']), int(s2['Y'])

                if 0 <= x < w and 0<= y < h:
                    sp_id = labels[y][x]
                    y_gt[sp_id] = label_id


        # print(333)
        labels = np.asarray( [[j if j not in y_gt else y_gt[j] for j in i] for i in labels])


        mask = self.mask_trans(mask)
        image = self.image_trans(image)
        labels = self.label_trans(labels)

        return (image, mask, labels, edges, len(unique_ids), y_gt)

    def getpolygons(self, index):
        t = self.train_dataset.images[index]
        t = t.replace('VOC2012/JPEGImages', 'scribble_annotation/pascal_2012')\
            .replace('jpg','xml')
        with open(t, 'r') as f:
            t = f.read()
        t = xmltodict.parse(t)['annotation']['polygon']
        return t

    def __len__(self):
        return len(self.train_dataset.images)


train_dataset = myVOC2012()
# train_loader = DataLoader(train_dataset, batch_size, shuffle=False,
#                           num_workers=0, pin_memory=False)  # Change num_workers to 16 in HDD machine

logger.info("voc 2012train set is successfully loaded")
