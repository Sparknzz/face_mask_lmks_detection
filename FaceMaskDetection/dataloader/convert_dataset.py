from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import shutil
import pathlib
import random

import PIL
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
import io
from collections import namedtuple, OrderedDict

supported_fmt = ['.jpg', '.JPG', '.png', '.PNG']

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def xml_to_csv(data_list):
    """
    将data_list表示的(图片, 标签)对转换成pandas.Dataframe记录
    """
    xml_list = []
    for data in data_list:

        tree = ET.parse(data['label'])
        root = tree.getroot()
        try:
            img = Image.open(data['image'])
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            print(f'打开{data["image"]}出错!')
            continue
        width, height = img.size
        img.close()

        for member in root.findall('object'):

            bndbox = member.find('bndbox')

            value = (data['image'],
                     width,
                     height,
                     member[0].text,
                     int(bndbox[0].text),
                     int(bndbox[1].text),
                     int(bndbox[2].text),
                     int(bndbox[3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

if __name__ == '__main__':

    dataset_dir='/home/data/FaceMaskDetection/val'
    annotations_dir = '/root/face_mask_lmks_detection/FaceMaskDetection/annos'

    dataset_path = pathlib.Path(dataset_dir)

    found_data_list = []

    for xml_file in dataset_path.glob('**/*.xml'):
        possible_images = [xml_file.with_suffix(suffix) for suffix in supported_fmt]
        supported_images = list(filter(lambda p: p.is_file(), possible_images))

        if len(supported_images) == 0:
            print(f'找不到对应的图片文件：`{xml_file.as_posix()}`')
            continue

        found_data_list.append({'image': supported_images[0], 'label': xml_file})

    random.shuffle(found_data_list) # 7971

    # 将XML转换成CSV格式
    train_xml_df = xml_to_csv(found_data_list)
    train_xml_df.to_csv(os.path.join(annotations_dir, 'val_labels.csv'), index=False)
    
    print('Successfully converted xml to csv.')