from torch.backends import cudnn

from utils import visualizer
from utils.conf import conf
from utils.logger import logger
from utils.visualizer import show_my_result

from voc2012 import *
from models.experiment_model_1 import model

import torch
import torch.nn as nn
from torch.optim.adam import Adam

"""
.___  ___.   ______    _______   _______  __      
|   \/   |  /  __  \  |       \ |   ____||  |     
|  \  /  | |  |  |  | |  .--.  ||  |__   |  |     
|  |\/|  | |  |  |  | |  |  |  ||   __|  |  |     
|  |  |  | |  `--'  | |  '--'  ||  |____ |  `----.
|__|  |__|  \______/  |_______/ |_______||_______|
"""
model = model()
loss = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=1e-3)
if conf.level == 'DEBUG':
    logger.info(model)
logger.info("model, loss, optim is ready")


"""
.______   .______       _______ .______      ___      .______       _______ 
|   _  \  |   _  \     |   ____||   _  \    /   \     |   _  \     |   ____|
|  |_)  | |  |_)  |    |  |__   |  |_)  |  /  ^  \    |  |_)  |    |  |__   
|   ___/  |      /     |   __|  |   ___/  /  /_\  \   |      /     |   __|  
|  |      |  |\  \----.|  |____ |  |     /  _____  \  |  |\  \----.|  |____ 
| _|      | _| `._____||_______|| _|    /__/     \__\ | _| `._____||_______|
"""
# cudnn.benchmark = True
if conf.multi_gpu:
    model = nn.DataParallel(model)
model = model.cuda()

loss = loss.cuda()
logger.info('model is ready to train / infer')


"""
.___________..______          ___       __  .__   __. 
|           ||   _  \        /   \     |  | |  \ |  | 
`---|  |----`|  |_)  |      /  ^  \    |  | |   \|  | 
    |  |     |      /      /  /_\  \   |  | |  . `  | 
    |  |     |  |\  \----./  _____  \  |  | |  |\   | 
    |__|     | _| `._____/__/     \__\ |__| |__| \__| 
"""
for (image, mask, labels, edges, sp_size) in train_dataset:
    (image, mask, labels, edges, sp_size) = (image.cuda(), mask.cuda(), labels.cuda(), edges, sp_size)
    # show_my_result(image, mask, labels)
    # print(edges)
    image = image.unsqueeze(0)
    logits = model(image, labels, edges, sp_size)
    print(logits)

    break

