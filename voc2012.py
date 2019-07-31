from utils.conf import conf
from utils.logger import logger

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets


batch_size = conf.batch_size # Todo

normalize = transforms.Normalize(mean=[.485, .456, .406],
                                 std=[.229, .224, .225])

train_dataset = datasets.VOCSegmentation('./',
                                         image_set='train',
                                         transform=transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize]))

train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                          num_workers=0, pin_memory=True) # Change num_workers to 16 in HDD machine

logger.info("voc 2012train set is successfully loaded")
