# generate mask landmarks as the original label file is .mat format
# convert .mat to .txt first

import scipy.io
import numpy as np
import pandas as pd
data = scipy.io.loadmat("/home/data/MAFA/MAFA-Label-Train/LabelTrainAll.mat")

# (
# array(['add_11.jpg'], dtype='<U10'), 
# array(['train_00000003.jpg'], dtype='<U18'), 
# array([[ 56, 170, 185, 185, 140, 198, 196, 208,  41,  56, 147, 182,   1, 3,   1,   1,   4,  -1,  -1,  -1,  -1]], 
# dtype=int16)
# ),

all_labels = data['label_train'][0] # all images labels

with open('mafa.txt', 'w+') as f:

    for per_img in all_labels:
        # per_img[0]
        img_name = per_img[1][0]
        img_labels = per_img[2]

        for label in img_labels:
            f_x1, f_y1, f_w, f_h = label[:4]
            p_x1, p_y1, p_x2, p_y2 = label[4:8]
            occ_type = label[12]

            # if f_w < 100 and f_h<100:
            #     continue

            if occ_type!=1:
                continue

            # as we want to combine mask and non-mask faces altogether, so let the other 3 landmarks to (-1)
            p_x3, p_y3, p_x4, p_y4, p_x5, p_y5 = 0, 0, 0, 0, 0, 0
            # don't need mask bboxes for now, so ignore them for this moment
            # m_x1, m_y1, m_w, m_h = label[8:12] # m_x1 m_y1 is relative position of f_x1 f_y1

            bbox = (f_x1, f_y1, f_w, f_h)
            landmark5 = (p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4, p_x5, p_y5)

            # [path to image]       [cls_label]{0, 1, 2}           [bbox_label]{ }      [landmark_label]
            # write all data as a row : eg [name f_x1, f_y1, f_w, f_h p_x1, p_y1, p_x2, p_y2, p_x3, p_y3, p_x4, p_y4, p_x5, p_y5)
            line = []

            line.append(str(img_name))

            for b in bbox:
                line.append(str(b))

            for landmark in landmark5:
                line.append(str(landmark))
            
            line.append('\n')

            line_str = ' '.join(line)
            f.write(line_str)
