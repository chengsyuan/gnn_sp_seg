from skimage import segmentation
import numpy as np
import cv2

def superpixel(cv2_img, debug = True):
    """
    parameters like 1111 and 10 are selected by my intuition
    :param cv2_img:  cv2.imread('../example_images/2007_000039.jpg')
    :param debug:    print debug info
    :return: labels: [h,w] numpy array, unique_ids: [0, 1, ..., superpixel_n - 1]
    """

    labels = segmentation.slic(cv2_img, n_segments=1111, compactness=10)
    unique_ids = np.unique(labels)

    if debug:
        print('this image has {} unique superpixels'.format(len(unique_ids)))

    return labels, unique_ids

def rend_superpixel(labels, random_seed = 888):
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

if __name__ == '__main__':
    cv2_img = cv2.imread('../example_images/2007_000123.jpg')

    labels, unique_ids = superpixel(cv2_img, debug=True)
    im_target_rgb = rend_superpixel(labels)

    cv2.imshow('image', cv2_img)
    cv2.imshow('superpixel_image', im_target_rgb)
    cv2.waitKey(0)