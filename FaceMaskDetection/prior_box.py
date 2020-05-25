import torch
from itertools import product as product
import numpy as np
from math import ceil
import cv2

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = [cfg['image_size'], cfg['image_size']]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])): # column i width j
                f_k = self.image_size[0] / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                s_k = self.min_sizes[k][0] # wrt original image
                mean += [cx, cy, s_k, s_k] # cx cy w h

                s_k_prime = self.min_sizes[k][1]
                mean += [cx, cy, s_k_prime, s_k_prime] # 2 prior box

                # rest of aspect ratios
                for ar in self.aspect_ratios: # 5 prior box note here, only for first anchor, not for the large anchor, 7 anchors in total
                    mean += [cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        if 0:
            empty_img = np.ones((640, 640, 3))

            for i in range(len(output)):
                cv2.rectangle(empty_img, (int(output.numpy()[i][0] * 640), int(output.numpy()[i][1] * 640), 
                int(output.numpy()[i][2] * 640), int(output.numpy()[i][3] * 640)), color=(0, 255, 0))
            
            cv2.imwrite('res.jpg', empty_img)


        if self.clip:
            output.clamp_(max=1, min=0)
            
        return output


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]: # why it is 2. [0.05, 0.075]  what is 0.05, this is the anchor width, use the width to generate same height anchor
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    
    return anchor_bboxes


if __name__ == '__main__':
    # anchors = generate_anchors(feature_map_sizes=[[40,40], [20,20]], anchor_sizes=[[0.05, 0.075], [0.1, 0.15]], anchor_ratios=[[1, 0.5], [1, 0.5]])

    cfg = {
        'name': 'mobilenet',
        'min_sizes': [[0.02, 0.0625], [0.0725, 0.175], [0.25, 0.6]], # 0.02 12 pixel 
        'aspect_ratios': [0.54, 0.63, 0.72],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'image_size': 640,
    }

    prior = PriorBox(cfg)
    prior.forward()