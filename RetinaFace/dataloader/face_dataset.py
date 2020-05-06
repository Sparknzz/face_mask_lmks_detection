import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class FaceDataset(data.Dataset):
    def __init__(self, txt_path, img_dirs, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.img_infos = []
        f = open(txt_path, 'r')
        lines = f.readlines()

        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.img_infos.append(labels_copy)
                    labels.clear()
                path = line[2:]
                if path.startswith('train_'):
                    fold_path = img_dirs[0]
                else:
                    fold_path = img_dirs[1]
                
                path = fold_path + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.img_infos.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.img_infos[index]

        annotations = np.zeros((0, 15))

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target