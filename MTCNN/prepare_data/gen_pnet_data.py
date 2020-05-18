import argparse
import numpy as np
import cv2
import os
import numpy.random as npr
import sys
import pandas as pd
from utils import calculate_iou
import xml.etree.ElementTree as ET

def assemble_pnet_data(anno_dir, output_file):

    net = 'pnet'

    with open(os.path.join(anno_dir, 'face_pos_12.txt'), 'r') as f:
        face_pos = f.readlines()

    with open(os.path.join(anno_dir, 'face_neg_12.txt'), 'r') as f:
        face_neg = f.readlines()

    with open(os.path.join(anno_dir, 'face_part_12.txt'), 'r') as f:
        face_part = f.readlines()

    with open(os.path.join(anno_dir, 'mask_pos_12.txt'), 'r') as f:
        mask_pos = f.readlines()

    with open(os.path.join(anno_dir, 'mask_neg_12.txt'), 'r') as f:
        mask_neg = f.readlines()

    with open(os.path.join(anno_dir, 'mask_part_12.txt'), 'r') as f:
        mask_part = f.readlines()

    with open(output_file, "w") as f:

        print(len(face_neg), len(face_pos), len(face_part), len(mask_neg), len(mask_part), len(mask_pos))
        ratio = [3, 1, 1]        

        # mask_pos_keep = npr.choice(len(mask_pos), size=base_num, replace=True)
        # mask_part_keep = npr.choice(len(mask_part), size=base_num, replace=True)
        # mask_neg_keep = npr.choice(len(mask_neg), size= base_num * 3, replace=True)

        # basically the pos cls including face and mask, so we have to make them same.
        pos = face_pos + mask_pos * 3
        part = face_part + mask_part
        neg = face_neg + mask_neg

        # 52000 for face_pos, 14000 for mask_pos
        # so we will use all face_pos, but 3 times of mask_pos. then will genenrate 100k+ pos data
        base_num = 100000
        #shuffle the order of the initial data
        pos_keep = npr.choice(len(pos), size=base_num)
        part_keep = npr.choice(len(part), size=base_num, replace=True)
        neg_keep = npr.choice(len(neg), size= base_num * 3, replace=True)


        # write the data according to the shuffled order
        for i in pos_keep:
            f.write(pos[i])

        for i in neg_keep:
            f.write(neg[i])

        for i in part_keep:
            f.write(part[i])


def gen_pnet_data(im_dir, data_dir, anno_file, label):
    # note as for small training purpose, only get the face greater than 40
    # one more important thing is when generating mask faces, it will generates lots of the non masked faces for negative, so when training mtcnn,
    # just ignore the negative data from mask/negative. just use positive would be fine
    neg_save_dir =  os.path.join(data_dir, "12/%s/negative"%(label))
    pos_save_dir =  os.path.join(data_dir, "12/%s/positive"%(label))
    part_save_dir = os.path.join(data_dir, "12/%s/part"%(label))

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    save_dir = os.path.join(data_dir, "pnet")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pos_save_file = os.path.join('../annos', '%s_pos_12.txt'%label)
    neg_save_file = os.path.join('../annos', '%s_neg_12.txt'%label)
    part_save_file = os.path.join('../annos', '%s_part_12.txt'%label)

    f1 = open(pos_save_file, 'w')
    f2 = open(neg_save_file, 'w')
    f3 = open(part_save_file, 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    
    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # don't care
    idx = 0

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        #image path
        im_name = annotation[0]
        #print(im_path)
        #boxed change to float type
        bbox = list(map(float, annotation[1:]))
        #gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        
        #load image
        im_path = os.path.join(im_dir, im_name + '.jpg')
        if not os.path.exists(im_path):
            im_path = os.path.join(im_dir, im_name + '.png')

        img = cv2.imread(im_path)

        idx += 1
        
        #if idx % 100 == 0:
            #print(idx, "images done")

        height, width, channel = img.shape

        neg_num = 0
        #1---->50
        # keep crop random parts, until have 50 negative examples
        # get 50 negative sample from every image
        while neg_num < 50:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = calculate_iou(crop_box, boxes)

            cropped_im = img[int(ny) : int(ny + size), int(nx) : int(nx + size), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)


            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)

                n_idx += 1
                neg_num += 1

        for box in boxes: # for each box we all do the postive collection in our dataset there are 39915 we may get 800k pos and part faces
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # generate negative examples that have overlap with gt, does this mean hard example ????
            for i in range(5):
                size = npr.randint(12,  min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue

                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = calculate_iou(crop_box, boxes)

                cropped_im = img[int(ny1) : int(ny1 + size), int(nx1) : int(nx1 + size), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                if w<5:
                    print (w)
                    continue

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0) # x1 + w / 2 means gt center_x, + delta_x represents box_center moving, size/2 means crop half size, 
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                
                if calculate_iou(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                
                elif calculate_iou(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))


    f1.close()
    f2.close()
    f3.close()

def txt2csv(mafa_anno_file):

    df = pd.read_table('mafa.txt', sep=' ',  header = None)
    df.columns=['filename', 'x1' , 'y1', 'w', 'h', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y', 'p5_x', 'p5_y', '']

    return df.iloc[:, :-1]

def combine_mafa_wider(mafa_anno_file, wider_anno_file):

    # convert txt to csv to combine them
    df_mafa = txt2csv(mafa_anno_file)

    df_mafa['label'] = 1

    df_wider = pd.read_csv(wider_anno_file)

    df_wider['label'] = 0

    df = pd.concat([df_mafa, df_wider], axis=0)

    df.to_csv('merged_data.csv', index=False)

def voc2txt(data_dir, select_cls):
    # select_cls is for set which cls data to generate. cls 1 for face, cls 2 for mask

    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    xml_files = list(filter(lambda x: x.endswith('.xml'), file_paths))

    _class_to_ind = dict(zip(('__background__', 'face', 'face_mask'), range(3)))

    if select_cls==1:
        anno_file = 'face.txt'
    else:
        anno_file = 'mask.txt'

    with open(anno_file, 'w') as f:

        for filename in xml_files:

            tree = ET.parse(filename)
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            # "Seg" area for pascal is just the box area
            # seg_areas = np.zeros((num_objs), dtype=np.float32)

            file_bboxes = []

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                cls = _class_to_ind[obj.find('name').text.lower().strip()]

                if cls==select_cls:
                    file_bboxes.append(str(x1))
                    file_bboxes.append(str(y1))
                    file_bboxes.append(str(x2))
                    file_bboxes.append(str(y2))

                # boxes[ix, :] = [x1, y1, x2, y2]
                
                # gt_classes[ix] = cls
                
                # overlaps[ix, cls] = 1.0
                
                # seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            # overlaps = scipy.sparse.csr_matrix(overlaps)

            if len(file_bboxes)==0:
                continue

            line = ' '.join(file_bboxes)

            line = filename.split('/')[-1].replace('.xml', '') + ' ' + line + '\n'

            f.write(line)

            # return {'boxes' : boxes,
            #         'gt_classes': gt_classes,
            #         'gt_overlaps' : overlaps,
            #         'flipped' : False,
            #         'seg_areas' : seg_areas}


if __name__ == '__main__':
    # mafa_anno_file = 'mafa.txt'
    # wider_anno_file = 'wider_face.csv'

    # combine_mafa_wider(mafa_anno_file, wider_anno_file)
    # voc2txt(data_dir='/home/data/train', select_cls=2)
    # gen_pnet_data('/home/data/train', '/home/data/my_data', 'face.txt', label='face')
    # gen_pnet_data('/home/data/train', '/home/data/my_data', 'mask.txt', label='mask')
    assemble_pnet_data(anno_dir='/root/face_mask_lmks_detection/MTCNN/annos', output_file = '/root/face_mask_lmks_detection/MTCNN/annos/pnet_train_imglist.txt')