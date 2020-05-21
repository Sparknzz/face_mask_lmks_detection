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

project_root = '/project'
os.makedirs('/project/dataset', exist_ok=True)
os.makedirs('/project/dataset/images', exist_ok=True)

train_data_dir = os.path.join(project_root, 'dataset/images/train/')
valid_data_dir = os.path.join(project_root, 'dataset/images/valid/')
annotations_dir = '/project/dataset/annotations'

supported_fmt = ['.jpg', '.JPG']

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def xml_to_csv(data_list):
    """将data_list表示的(图片, 标签)对转换成pandas.Dataframe记录
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
            value = (data['image'],
                     width,
                     height,
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(valid_data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    dataset_dir='/home/data/13'

    # 遍历数据集目录下所有xml文件及其对应的图片
    dataset_path = pathlib.Path(dataset_dir)
    found_data_list = []

    for xml_file in dataset_path.glob('**/*.xml'):
        possible_images = [xml_file.with_suffix(suffix) for suffix in supported_fmt]
        supported_images = list(filter(lambda p: p.is_file(), possible_images))
        if len(supported_images) == 0:
            print(f'找不到对应的图片文件：`{xml_file.as_posix()}`')
            continue

        found_data_list.append({'image': supported_images[0], 'label': xml_file})

    # 随机化数据集，将数据集拆分成训练集和验证集，并将其拷贝到/project/train/src_repo/dataset下
    random.shuffle(found_data_list)

    train_data_count = len(found_data_list) * 4 / 5
    train_data_list = []
    valid_data_list = []

    for i, data in enumerate(found_data_list):
        if i < train_data_count:  # 训练集
            dst = train_data_dir
            data_list = train_data_list
        else:  # 验证集
            dst = valid_data_dir
            data_list = valid_data_list

        image_dst = (pathlib.Path(dst) / data['image'].name).as_posix()
        label_dst = (pathlib.Path(dst) / data['label'].name).as_posix()
        shutil.copy(data['image'].as_posix(), image_dst)
        shutil.copy(data['label'].as_posix(), label_dst)
        data_list.append({'image': image_dst, 'label': label_dst})

    # 将XML转换成CSV格式
    train_xml_df = xml_to_csv(train_data_list)
    train_xml_df.to_csv(os.path.join(annotations_dir, 'train_labels.csv'), index=False)
    
    valid_xml_df = xml_to_csv(valid_data_list)
    valid_xml_df.to_csv(os.path.join(annotations_dir, 'valid_labels.csv'), index=False)
    print('Successfully converted xml to csv.')