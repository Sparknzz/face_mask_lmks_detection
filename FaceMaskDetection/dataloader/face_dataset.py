import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd

class FaceDataset(data.Dataset):
    def __init__(self, csv_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.img_infos = []
        self.df = pd.read_csv(csv_path)

        cls_dict = {'__background__': 0, 'face': 1, 'face_mask': 2}

        for filename, df_group in self.df.groupby('filename'):
            self.imgs_path.append(filename)
            info = []

            for i in range(len(df_group)):
                # 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'
                x1 = df_group.iloc[i]['xmin']
                y1 = df_group.iloc[i]['ymin']
                x2 = df_group.iloc[i]['xmax']
                y2 = df_group.iloc[i]['ymax']
                c = cls_dict[df_group.iloc[i]['class']]
                w = df_group.iloc[i]['width']
                h = df_group.iloc[i]['height']

                info.append([x1, y1, x2, y2, w, h, c])

            self.img_infos.append(info)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):

        img = cv2.imread(self.imgs_path[index])
        
        height, width, _ = img.shape

        labels = self.img_infos[index]

        annotations = np.zeros((0, 5))

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))

            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # x2
            annotation[0, 3] = label[3]  # y2
            
            annotation[0, 4] = int(label[-1]) # 1 is face, 2 is face_mask 

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target