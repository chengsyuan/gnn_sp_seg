from torch.backends import cudnn
from torch.optim import SGD, Adagrad

from utils import visualizer
from utils.conf import conf
from utils.logger import logger
from utils.visualizer import show_my_result, show_ans

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
optim = Adam(model.parameters())
if conf.level == 'DEBUG':
    logger.info(model)
logger.info("model, loss, optim is ready")

for param in model.features.parameters():
    param.requires_grad = False

"""
.______   .______       _______ .______      ___      .______       _______ 
|   _  \  |   _  \     |   ____||   _  \    /   \     |   _  \     |   ____|
|  |_)  | |  |_)  |    |  |__   |  |_)  |  /  ^  \    |  |_)  |    |  |__   
|   ___/  |      /     |   __|  |   ___/  /  /_\  \   |      /     |   __|  
|  |      |  |\  \----.|  |____ |  |     /  _____  \  |  |\  \----.|  |____ 
| _|      | _| `._____||_______|| _|    /__/     \__\ | _| `._____||_______|
"""
cudnn.benchmark = True
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

for epoch in range(1, 1 + 100):
    logger.critical('Epoch:{}'.format(epoch))

    for idx, (image, mask, labels, edges, sp_size, y_gt) in enumerate(train_dataset):

        (image, mask, labels, edges, sp_size) = (image.cuda(), mask.cuda(), labels.cuda(), edges, sp_size)
        image = image.unsqueeze(0)
        while True:
            logits = model(image, labels, edges, sp_size)

            selected_sp = list(y_gt.keys())
            logits_ = logits[selected_sp]
            ans_ = torch.LongTensor([y_gt[i] for i in selected_sp]).cuda()

            _, ids = torch.topk(logits_, 1)
            print(selected_sp,
                  logits.detach().cpu().numpy().tolist(),
                  ids.detach().cpu().numpy().tolist(),
                  ans_.detach().cpu().numpy().tolist(), sep='\n\n\n')

            _, ids = torch.topk(logits, 1)
            ans = ids.cpu().numpy().tolist()
            ans_labels = labels.cpu().numpy()

            for idx, v in enumerate(ans):
                ans_labels[ans_labels == idx] = v


            l = loss(logits_[ans_==0], ans_[ans_==0]) + \
                1.5 * loss(logits_[ans_!=0], ans_[ans_!=0])

            # l = loss(logits_, ans_)

            loss_item = l.item()



            _, predicted = torch.max(logits_.data, 1)
            acc_item = 100. * (predicted == ans_).sum().item() / ans_.size(0)

            if loss_item < 1. and acc_item > 90.:
                show_ans(ans_labels)

            logger.info('{} of {}: {} {}%'.format(idx, len(train_dataset), loss_item, acc_item))

            optim.zero_grad()
            l.backward()
            optim.step()

            torch.cuda.empty_cache()

        # show_my_result(image, mask, labels)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, './ckpts/{}'.format(epoch))


