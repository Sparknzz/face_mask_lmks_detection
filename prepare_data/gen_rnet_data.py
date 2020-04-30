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


def detect_pnet(pnet, im, thresh, min_face_size=50, scale_factor=0.709):
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

        if len(boxes)!=0:
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


def t_net(model_path, batch_size=1, test_mode="PNet",
             thresh=[0.6, 0.6, 0.7], min_face_size=40,
             stride=2, slide_window=False, shuffle=False, vis=False):
    
    # load pnet model
    if model_path is not None:
        pnet = PNet()
        pnet.load_state_dict(torch.load(model_path))
        pnet.cuda()
        pnet.eval()

    boxes_align = np.array([])
    landmark_align =np.array([])


    im = cv2.imread('/root/face_mask_lmks_detection/test.jpg')
 
    t = time.time()


    with torch.no_grad():
        boxes_align = detect_pnet(pnet, im, thresh[0])
    
    if boxes_align is None:
        return np.array([]), np.array([])

    t1 = time.time() - t


    for box in boxes_align:
        if box[5]==1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    cv2.imwrite('result.jpg', im)



if __name__ == '__main__':
    image_size = 24
    base_dir = '/home/data/cleaned_data'
    data_dir = '/home/data/my_mtcnn_data/24'
    
    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    t_net('/root/face_mask_lmks_detection/weights/pnet_epoch_15.pth')
