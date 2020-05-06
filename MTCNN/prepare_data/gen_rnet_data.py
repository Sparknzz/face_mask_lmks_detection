import cv2
import numpy as np
import time
import os
import torch
import sys
sys.path.append(os.path.realpath(__file__).replace('/prepare_data/gen_rnet_data.py',''))
import image_tools
from model.pnet import PNet 
from utils import *
from dataloader.image_reader import TestLoader

def resize_image(img, scale):
    """
        resize image and transform dimention to [batchsize, channel, height, width]
    Parameters:
    ----------
        img: numpy array , height x width x channel
            input image, channels in BGR order here
        scale: float number
            scale factor of resize operation
    Returns:
    -------
        transformed image tensor , 1 x channel x height x width
    """
    height, width, channels = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

def detect_pnet(pnet, im, thresh, min_face_size=100, scale_factor=0.709):
    h, w, c = im.shape
    net_size = 12

    current_scale = float(net_size) / min_face_size 

    im_resized = resize_image(im, current_scale)
    
    current_height, current_width, _ = im_resized.shape

    all_boxes = list()
    while min(current_height, current_width) > net_size:
        feed_imgs = []
        image_tensor = image_tools.convert_image_to_tensor(im_resized)
        feed_imgs.append(image_tensor)
        feed_imgs = torch.stack(feed_imgs)
        feed_imgs = feed_imgs.float().cuda()

        cls_map, reg = pnet(feed_imgs) # (batch, 2, h, w), (batch, 4*2, h, w)
        
        # remove the batch dimension, for batch is always 1 for inference.
        cls_probability = cls_map.cpu().numpy()[0]
        reg_np = reg.cpu().numpy()[0]

        boxes = generate_bounding_box(cls_probability.transpose(1,2,0), reg_np.transpose(1,2,0), current_scale, thresh) # num, 10

        if len(boxes) != 0:
            all_boxes.append(boxes)

        current_scale *= scale_factor
        im_resized = resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape


    if len(all_boxes) == 0:
        return None, None

    all_boxes = np.concatenate(all_boxes, axis=0)

    bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

    align_topx = all_boxes[:, 0] + all_boxes[:, 4] * bw
    align_topy = all_boxes[:, 1] + all_boxes[:, 5] * bh
    align_bottomx = all_boxes[:, 2] + all_boxes[:, 6] * bw
    align_bottomy = all_boxes[:, 3] + all_boxes[:, 7] * bh

      # refine the boxes
    boxes_align = np.vstack([align_topx,
                            align_topy,
                            align_bottomx,
                            align_bottomy, all_boxes[:, 8], all_boxes[:, 9]])

    boxes_align = boxes_align.T

    keep = nms(boxes_align, 0.5, 'Union')

    return boxes_align[keep]

def save_hard_example(net, data, save_path):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    im_idx_list = data['images']
    # print(images[0])
    gt_boxes_list = data['bboxes']

    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    # save files
    neg_label_file = "../annos/neg_%d.txt" % (image_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "../annos/pos_%d.txt" % (image_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "../annos/part_%d.txt" % (image_size)
    part_file = open(part_label_file, 'w')

    import pickle
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)

        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                #save the examples
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)
                # print(save_file)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(model_path, batch_size=1, test_mode="PNet",
             thresh=[0.6, 0.6, 0.7], stride=2, slide_window=False, shuffle=False, vis=False):
    
    # load pnet model
    if model_path is not None:
        pnet = PNet()
        pnet.load_state_dict(torch.load(model_path))
        pnet.cuda()
        pnet.eval()

    #########################################################################
    if 1:
        im = cv2.imread('/root/face_mask_lmks_detection/test.jpg')
    
        t = time.time()

        boxes_align = np.array([])
        landmark_align =np.array([])

        with torch.no_grad():
            boxes_align = detect_pnet(pnet, im, thresh[0])
        
        if boxes_align is None:
            return np.array([]), np.array([])

        t1 = time.time() - t


        for box in boxes_align:
            if box[5]==1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

        cv2.imwrite('result.jpg', im)

    ####################################################################################
    # data = read_annotation(wider_base_dir, label_path = './wider_face_train_bbx_gt.txt')
    # test_data = TestLoader(data['images'])
    # all_boxes = []
    # sum_time = 0

    # for databatch in test_data:
    #     st = time.time()
    #     boxes_align = detect_pnet(pnet, im, thresh[0])
    #     t1 = time.time() - st
    #     sum_time += t1
    #     all_boxes.append(boxes_align)

    # print('num of images', test_data.size)
    # print("time cost in average" + '{:.3f}'.format(sum_time/test_data.size))
    # ########################################################################################
    # import pickle
    # save_path = os.path.join(data_dir)
    # save_file = os.path.join(save_path, "detections.pkl")
    # with open(save_file, 'wb') as f:
    #     pickle.dump(all_boxes, f,1)
    
    # print("%s starting OHEM " % image_size)
    # save_hard_example(image_size, data, save_path)


if __name__ == '__main__':
    image_size = 24
    base_dir = '/home/data/cleaned_data'
    data_dir = '/home/data/my_mtcnn_data/24'
    wider_base_dir = '/home/data/detection'

    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    t_net('/root/face_mask_lmks_detection/weights/pnet_epoch_16.pth')
