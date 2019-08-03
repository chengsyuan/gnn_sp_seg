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

logger.info('torchvsion {}'.format(torchvision.__version__))

batch_size = conf.batch_size


class myVOC2012(Dataset):

    def __init__(self):
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

        # transform them to tensors

        labels, unique_ids = superpixel(image, debug=True)
        # print(labels)
        
        edges = superpixel_edge_list(labels)
        mask = self.mask_trans(mask)
        image = self.image_trans(image)
        labels = self.label_trans(labels)


        return (image, mask, labels, edges)

    def __len__(self):
        return len(self.train_dataset.images)


train_dataset = myVOC2012()
# train_loader = DataLoader(train_dataset, batch_size, shuffle=False,
#                           num_workers=0, pin_memory=False)  # Change num_workers to 16 in HDD machine

logger.info("voc 2012train set is successfully loaded")
