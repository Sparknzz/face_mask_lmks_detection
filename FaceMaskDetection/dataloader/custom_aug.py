import cv2
import numpy as np
import random

def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    a is boxes, batch, 4.   b is croped img
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)

        roi = np.array((l, t, l + w, t + h))

        # to make sure croped image contains bbox
        value = matrix_iof(boxes, roi[np.newaxis])

        flag = (value >= 1)

        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        # todo for me its better to calculate the croped face and original roi. cos lots of those croped face would be half face. 
        # if those are half face, then label it -1.

        boxes_t = boxes[mask_a].copy() # keep gt boxes
        labels_t = labels[mask_a].copy() # labels_t still including -1

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]] # H W

        # crop bbox
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]

        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

	    # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim

        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag

    return image, boxes, labels, pad_image_flag

def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        
    return image, boxes


def _resize_subtract_mean(image, insize, rgb_mean):
    # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image -= rgb_mean

    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy() # 1 2 -1, -1 means ignored face, 1 means positive

        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)

        image_t, boxes_t = _mirror(image_t, boxes_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1) # 1, 1
        targets_t = np.hstack((boxes_t, labels_t)) # 1, 4 + 1

        return image_t, targets_t