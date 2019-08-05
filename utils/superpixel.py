from utils.conf import conf
from utils.logger import logger

from skimage import segmentation
from skimage.future import graph
import numpy as np
import cv2

def superpixel(cv2_img, n_segments=1111,
               compactness = 10, normalized_cut = False, debug = True):
    """
    parameters like 1111 and 10 are selected by my intuition
    :param cv2_img:  cv2.imread('../example_images/2007_000039.jpg')
    :param debug:    print debug info
    :return: labels: [h,w] numpy array, unique_ids: [0, 1, ..., superpixel_n - 1]
    """

    # labels = segmentation.slic(cv2_img, n_segments=1111, compactness=10)
    labels = segmentation.slic(cv2_img,
                               n_segments=n_segments,
                               compactness=compactness)

    if normalized_cut:
        g = graph.rag_mean_color(cv2_img, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)

    unique_ids = np.unique(labels)

    logger.debug('this image has {} unique superpixels'.format(len(unique_ids)))

    return labels, unique_ids

def rend_superpixel(labels, random_seed = conf.init_seed):
    """
    input the labels of the picture, return a [h, w, 3] RGB image for visualization
    :param labels:       [h, w] numpy array
    :param random_seed:  default is 888 for reproduce
    :return:
    """
    np.random.seed(random_seed)
    label_colours = np.random.randint(255, size=(100, 3))

    im_target_rgb = np.array([label_colours[c % 100] for c in labels])
    im_target_rgb = im_target_rgb.reshape((labels.shape[0], labels.shape[1], 3)).astype(np.uint8)

    return im_target_rgb

def superpixel_edge_list(labels):
    """
    get edge list given the superpixel labels
    :param labels: [h, w] numpy array
    :return: bneighbors [[start_nodes], [end_nodes]]
    """
    # get the right/below adjacent node by shift right/down
    vs_right = np.vstack([labels[:, :-1].ravel(), labels[:, 1:].ravel()])
    vs_below = np.vstack([labels[:-1, :].ravel(), labels[1:, :].ravel()])

    # reverse the edge list to get a bidirectional edge list
    bneighbors = np.unique(np.hstack([vs_right, np.flip(vs_right, axis=0),
                                      vs_below, np.flip(vs_below, axis=0)]), axis=1)

    # mask self-to-self edges
    mask = np.not_equal(bneighbors[0], bneighbors[1])
    bneighbors = bneighbors[:, mask]

    return bneighbors

if __name__ == '__main__':
    cv2_img = cv2.imread('../example_images/2007_000032.jpg')

    labels, unique_ids = superpixel(cv2_img, normalized_cut=True, debug=True)
    im_target_rgb = rend_superpixel(labels)
    edgelist = superpixel_edge_list(labels)
    logger.debug(edgelist)

    cv2.imshow('image', cv2_img)
    cv2.imshow('superpixel_image', im_target_rgb)
    cv2.waitKey(0)