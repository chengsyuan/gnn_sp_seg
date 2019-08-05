from utils.conf import conf
from utils.logger import logger

import numpy as np
import cv2

def show_my_result(image, masks, labels, ans_labels, random_seed = conf.init_seed):

    np.random.seed(random_seed)
    label_colours = np.random.randint(255, size=(100, 3))

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.show()

    im_target_rgb = np.array([label_colours[c % 100] for c in masks])
    im_target_rgb = im_target_rgb.reshape((masks.shape[1], masks.shape[2], 3)).astype(np.uint8)
    plt.imshow(im_target_rgb)
    plt.show()

    im_target_rgb = np.array([label_colours[c % 100] for c in labels])
    im_target_rgb = im_target_rgb.reshape((labels.shape[1], labels.shape[2], 3)).astype(np.uint8)
    plt.imshow(im_target_rgb)
    plt.show()

    im_target_rgb = np.array([label_colours[c % 100] for c in ans_labels])
    im_target_rgb = im_target_rgb.reshape((ans_labels.shape[1], ans_labels.shape[2], 3)).astype(np.uint8)
    plt.imshow(im_target_rgb)
    plt.show()

def show_ans(ans_labels, random_seed = conf.init_seed):
    np.random.seed(random_seed)
    label_colours = np.random.randint(255, size=(100, 3))

    import matplotlib.pyplot as plt

    im_target_rgb = np.array([label_colours[c % 100] for c in ans_labels])
    im_target_rgb = im_target_rgb.reshape((ans_labels.shape[1], ans_labels.shape[2], 3)).astype(np.uint8)
    plt.imshow(im_target_rgb)
    plt.show()