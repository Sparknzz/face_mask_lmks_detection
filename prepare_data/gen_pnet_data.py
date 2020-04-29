import argparse
import numpy as np
import cv2
import os
import numpy.random as npr
import sys
import pandas as pd
sys.path.append(os.path.realpath(__file__).replace('/prepare_data/gen_pnet_data.py',''))
print(sys.path)
from utils import calculate_iou


def assemble_pnet_data(anno_dir, output_file):

    anno_list = []

    # pnet_landmark_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_LANDMARK_ANNO_FILENAME)
    pnet_postive_file = os.path.join(anno_dir, 'face_pos_12.txt')
    # pnet_part_file = os.path.join(anno_dir, 'face_part_12.txt')
    pnet_neg_file = os.path.join(anno_dir, 'face_neg_12.txt')

    pnet_postive_file_mask = os.path.join(anno_dir, 'mask_pos_12.txt')
    # pnet_part_file_mask = os.path.join(anno_dir, 'mask_part_12.txt')

    # here we only combine face pos, mask pos, face neg, let the label be 0(non face), 1(face), 2(mask)
    anno_list.append(pnet_postive_file) # lable 1
    anno_list.append(pnet_postive_file_mask) # label 1
    anno_list.append(pnet_neg_file) # label 0

    for anno_file in anno_list:

        with open(anno_file, 'r') as f:
            anno_lines = f.readlines()

        base_num = 250000

        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)

        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        
        chose_count = 0
        
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                image_file = anno_lines[idx].split(' ')[0]
                
                if not os.path.exists(image_file):
                    continue

                if anno_file == pnet_postive_file_mask:
                    line = anno_lines[idx].split(' ')
                    line[1] = str(2)
                    new_line = ' '.join(line)
                    f.write(new_line)
                else:
                    f.write(anno_lines[idx])
                chose_count+=1

    print("PNet train annotation result file path: %s" % output_file)




def gen_pnet_data_pandas(data_dir, anno_file, prefix):
    # note as for small training purpose, only get the face greater than 40
    # one more important thing is when generating mask faces, it will generates lots of the non masked faces for negative, so when training mtcnn,
    # just ignore the negative data from mask/negative. just use positive would be fine
    neg_save_dir =  os.path.join(data_dir, "12/face/negative")
    pos_save_dir =  os.path.join(data_dir, "12/face/positive")
    part_save_dir = os.path.join(data_dir, "12/face/part")

    mask_neg_save_dir =  os.path.join(data_dir, "12/mask/negative")
    mask_pos_save_dir =  os.path.join(data_dir, "12/mask/positive")
    mask_part_save_dir = os.path.join(data_dir, "12/mask/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir, mask_neg_save_dir, mask_pos_save_dir, mask_part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    save_dir = os.path.join(data_dir, "pnet")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pos_save_file = os.path.join('../annos', 'face_pos_12.txt')
    neg_save_file = os.path.join('../annos', 'face_neg_12.txt')
    part_save_file = os.path.join('../annos', 'face_part_12.txt')

    mask_pos_save_file = os.path.join('../annos', 'mask_pos_12.txt')
    mask_neg_save_file = os.path.join('../annos', 'mask_neg_12.txt')
    mask_part_save_file = os.path.join('../annos', 'mask_part_12.txt')

    f1 = open(pos_save_file, 'w')
    f2 = open(neg_save_file, 'w')
    f3 = open(part_save_file, 'w')

    f4 = open(mask_pos_save_file, 'w')
    f5 = open(mask_neg_save_file, 'w')
    f6 = open(mask_part_save_file, 'w')

    df = pd.read_csv(anno_file)
    df_40 = df[(df['w']>40) & (df['h']>40)]
    groups = df_40.groupby('filename')

    num = len(groups)
    print("%d pics in total" % num)
    p_idx = 0
    n_idx = 0
    d_idx = 0
    idx = 0

    for filename, file_df in groups:
        if filename.endswith('.jpg'):
            im_path = os.path.join(prefix, filename)
        else:
            im_path = os.path.join(prefix, filename+'.jpg')

        x1 = np.array(file_df.loc[:, 'x1']).reshape(len(file_df), 1)
        y1 = np.array(file_df.loc[:, 'y1']).reshape(len(file_df), 1)
        w = np.array(file_df.loc[:, 'w']).reshape(len(file_df), 1)
        h = np.array(file_df.loc[:, 'h']).reshape(len(file_df), 1)

        label = file_df.loc[:, 'label'].tolist()[0]

        boxes = np.concatenate([x1, y1, x1+w, y1+h], axis=-1)
        # boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)

        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print(idx, "images done")

        height, width, channel = img.shape

        neg_num = 0

        while neg_num < 50:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            Iou = calculate_iou(crop_box, boxes)

            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3

                if label:
                    save_file = os.path.join(mask_neg_save_dir, "%s.jpg"%n_idx)
                    f5.write(save_file + ' 0\n')
                else:
                    save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                    f2.write(save_file + ' 0\n')
                
                cv2.imwrite(save_file, resized_im)

                if 0:
                    cv2.imwrite(save_file.replace('.jpg', '_'+str(n_idx)+'.jpg'), cropped_im)
                
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

                cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
                
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    if label:
                        save_file = os.path.join(mask_neg_save_dir, "%s.jpg"%n_idx)
                        f5.write(save_file + ' 0\n')
                    else:
                        save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                        f2.write(save_file + ' 0\n')

                    cv2.imwrite(save_file, resized_im)
                    if 0:
                        cv2.imwrite(save_file.replace('.jpg', '_'+str(n_idx)+'.jpg'), cropped_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
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
                    
                    if label:
                        save_file = os.path.join(mask_pos_save_dir, "%s.jpg"%p_idx)
                        f4.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    else:
                        save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                        f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    cv2.imwrite(save_file.replace('.jpg', '_'+str(p_idx)+'.jpg'), cropped_im)
                    p_idx += 1
                
                elif calculate_iou(crop_box, box_) >= 0.4:
                    if label:
                        save_file = os.path.join(mask_part_save_dir, "%s.jpg"%d_idx)
                        f6.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    else:
                        save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                        f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))

                    cv2.imwrite(save_file, resized_im)
                    if 0:
                        cv2.imwrite(save_file.replace('.jpg', '_'+str(d_idx)+'.jpg'), cropped_im)
                    d_idx += 1
            
            print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()

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

if __name__ == '__main__':

    # traindata_store = '../dataset'
    # annotation_file = 'wider_face_train.txt' # note for pnet only cal bbox,  so only concern about bbox data
    # prefix_path = '/home/data/detection/WIDER_train/images' # define the image stored root
    # gen_pnet_data(traindata_store, annotation_file, prefix_path)


    # mafa_anno_file = 'mafa.txt'
    # wider_anno_file = 'wider_face.csv'

    # combine_mafa_wider(mafa_anno_file, wider_anno_file)

    # gen_pnet_data_pandas('/home/data/my_mtcnn_data', anno_file='./merged_data.csv',prefix='/home/data/cleaned_data')
    # after get all the images, shuffle the img list

    assemble_pnet_data(anno_dir='/root/face_mask_lmks_detection/annos', output_file = '/root/face_mask_lmks_detection/annos/pnet_train_imglist.txt')

