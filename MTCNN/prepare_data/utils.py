import numpy as np

def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)

    data['images'] = images#all images
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

def calculate_iou(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr

def generate_bounding_box(probability, reg, scale, threshold):
    """
        generate bbox from feature map
    Parameters:
    ----------
        map: numpy array ,        n, m, 2
            detect score for each position
        reg: numpy array ,        n, m, 2 * 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """

    h, w, c = probability.shape
    
    stride = 2 # cos the img is pooling onece in pnet
    cellsize = 12

    # based on probability filter the cls bbox6
    idx = np.where(probability > threshold) # h, w, 2
    reg = reg.reshape(h, w, c, 4)

    dx1, dy1, dx2, dy2 = [reg[idx[0], idx[1], idx[2], i] for i in range(4)]
    reg = np.array([dx1, dy1, dx2, dy2])
    score = probability[idx[0], idx[1], idx[2]]
    label = np.array(idx[2])
    
    # find nothing
    if idx[0].size == 0:
        return np.array([])

    boundingbox = np.vstack([np.round((stride * idx[1]) / scale), \
                                np.round((stride * idx[0]) / scale), \
                                np.round((stride * idx[1] + cellsize) / scale), \
                                np.round((stride * idx[0] + cellsize) / scale), \
                                reg, \
                                score, \
                                label
                                ])

    return boundingbox.T

def nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    # note we need a label match
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    scores = dets[:, 4]
    # labels = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]]) # eg bbox 10*6  now becomes 9, 

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        # label_match = labels[i] == labels[order[1:]] # 9,

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        large_overlap = ovr > thresh
        
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        # invalid = large_overlap & label_match # 9, 

        # inds = np.where(~invalid)[0]

        # order = order[inds + 1]

        inds = np.where(ovr <= thresh)[0]
        
        order = order[inds + 1]

    return keep